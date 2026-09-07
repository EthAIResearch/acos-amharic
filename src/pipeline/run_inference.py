"""
Stage 6c: full end-to-end pipeline inference + evaluation.

Loads every trained checkpoint (Stages 1, 3, 4, 5's up-to-9 sub-models),
runs the whole pipeline on the test set, assembles quads (assemble_quads.py),
and evaluates with exact-match F1 split by explicit/implicit (evaluate.py).

Usage:
    python run_inference.py --config ../../configs/stage6_pipeline.yaml

Requires all referenced checkpoints to already exist -- this is the final
stage, run only once Stages 1, 3, 4, and at least the Stage 5 detectors are
trained. Category/sentiment checkpoints for a given Stage 5 anchor are only
loaded if that anchor's config path is provided -- omit any you haven't
trained yet and this script will skip that sub-model's contribution
(sentences that would have needed it just won't get that quad, which
correctly shows up as lower recall in the results rather than crashing).
"""
import argparse
import json
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2_pairing"))
from candidates import heuristic_pairs
from inference_utils import load_pair_classifier, load_stage1_model, extract_spans, classify_span_pair
from assemble_quads import assemble_quads, SentencePredictions
from evaluate import evaluate_quads

SENTIMENT_ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
BINARY_ID2LABEL = {0: False, 1: True}


def load_all_models(cfg, device):
    models = {}

    models["stage1"] = load_stage1_model(cfg["stage1_ckpt"], device)

    with open(cfg["label_space"], encoding="utf-8") as f:
        categories = json.load(f)["categories"]
    category_id2label = {i: c for i, c in enumerate(categories)}

    models["stage3_category"] = load_pair_classifier(cfg["stage3_ckpt"], len(categories), device)
    models["stage4_sentiment"] = load_pair_classifier(cfg["stage4_ckpt"], 3, device)

    # Stage 5: each key is optional -- only load what's been trained so far
    optional_stage5 = {
        "detect_implicit_aspect": ("stage5_detect_implicit_aspect_ckpt", 2),
        "detect_implicit_opinion": ("stage5_detect_implicit_opinion_ckpt", 2),
        "detect_fully_implicit": ("stage5_detect_fully_implicit_ckpt", 2),
        "category_opinion_anchor": ("stage5_category_opinion_anchor_ckpt", len(categories)),
        "category_aspect_anchor": ("stage5_category_aspect_anchor_ckpt", len(categories)),
        "category_fully_implicit": ("stage5_category_fully_implicit_ckpt", len(categories)),
        "sentiment_opinion_anchor": ("stage5_sentiment_opinion_anchor_ckpt", 3),
        "sentiment_aspect_anchor": ("stage5_sentiment_aspect_anchor_ckpt", 3),
        "sentiment_fully_implicit": ("stage5_sentiment_fully_implicit_ckpt", 3),
    }
    for key, (cfg_key, num_labels) in optional_stage5.items():
        if cfg.get(cfg_key):
            models[f"stage5_{key}"] = load_pair_classifier(cfg[cfg_key], num_labels, device)
        else:
            models[f"stage5_{key}"] = None
            print(f"NOTE: {cfg_key} not provided -- sentences needing this sub-model "
                  f"will simply not get that quad (shows up as recall loss, not a crash).")

    return models, category_id2label


def run_pipeline_on_sentence(tokens, models, category_id2label, device, max_length=256):
    stage1_model, stage1_tok = models["stage1"]
    aspect_spans, opinion_spans = extract_spans(stage1_model, stage1_tok, tokens, device, max_length)

    explicit_pairs = heuristic_pairs(aspect_spans, opinion_spans)

    stage3_model, stage3_tok = models["stage3_category"]
    stage4_model, stage4_tok = models["stage4_sentiment"]
    pair_category, pair_sentiment = {}, {}
    matched_aspects, matched_opinions = set(), set()
    for a1, a2, o1, o2 in explicit_pairs:
        key = (a1, a2, o1, o2)
        matched_aspects.add((a1, a2))
        matched_opinions.add((o1, o2))
        pair_category[key] = classify_span_pair(stage3_model, stage3_tok, tokens, device,
                                                  category_id2label, (a1, a2), (o1, o2), max_length)
        pair_sentiment[key] = classify_span_pair(stage4_model, stage4_tok, tokens, device,
                                                  SENTIMENT_ID2LABEL, (a1, a2), (o1, o2), max_length)

    preds = SentencePredictions(
        aspect_spans=aspect_spans, opinion_spans=opinion_spans,
        explicit_pairs=explicit_pairs, pair_category=pair_category, pair_sentiment=pair_sentiment,
    )

    if models["stage5_detect_implicit_aspect"]:
        det_model, det_tok = models["stage5_detect_implicit_aspect"]
        for o_span in opinion_spans:
            if o_span in matched_opinions:
                continue
            is_implicit = classify_span_pair(det_model, det_tok, tokens, device, BINARY_ID2LABEL,
                                              aspect_span=None, opinion_span=o_span, max_length=max_length)
            preds.implicit_aspect_flag[o_span] = is_implicit
            if is_implicit and models["stage5_category_opinion_anchor"]:
                cm, ct = models["stage5_category_opinion_anchor"]
                preds.implicit_aspect_category[o_span] = classify_span_pair(
                    cm, ct, tokens, device, category_id2label, None, o_span, max_length)
            if is_implicit and models["stage5_sentiment_opinion_anchor"]:
                sm, st = models["stage5_sentiment_opinion_anchor"]
                preds.implicit_aspect_sentiment[o_span] = classify_span_pair(
                    sm, st, tokens, device, SENTIMENT_ID2LABEL, None, o_span, max_length)

    if models["stage5_detect_implicit_opinion"]:
        det_model, det_tok = models["stage5_detect_implicit_opinion"]
        for a_span in aspect_spans:
            if a_span in matched_aspects:
                continue
            is_implicit = classify_span_pair(det_model, det_tok, tokens, device, BINARY_ID2LABEL,
                                              aspect_span=a_span, opinion_span=None, max_length=max_length)
            preds.implicit_opinion_flag[a_span] = is_implicit
            if is_implicit and models["stage5_category_aspect_anchor"]:
                cm, ct = models["stage5_category_aspect_anchor"]
                preds.implicit_opinion_category[a_span] = classify_span_pair(
                    cm, ct, tokens, device, category_id2label, a_span, None, max_length)
            if is_implicit and models["stage5_sentiment_aspect_anchor"]:
                sm, st = models["stage5_sentiment_aspect_anchor"]
                preds.implicit_opinion_sentiment[a_span] = classify_span_pair(
                    sm, st, tokens, device, SENTIMENT_ID2LABEL, a_span, None, max_length)

    if models["stage5_detect_fully_implicit"]:
        det_model, det_tok = models["stage5_detect_fully_implicit"]
        preds.fully_implicit_flag = classify_span_pair(det_model, det_tok, tokens, device, BINARY_ID2LABEL,
                                                         None, None, max_length)
        if preds.fully_implicit_flag:
            if models["stage5_category_fully_implicit"]:
                cm, ct = models["stage5_category_fully_implicit"]
                preds.fully_implicit_category = classify_span_pair(cm, ct, tokens, device, category_id2label,
                                                                     None, None, max_length)
            if models["stage5_sentiment_fully_implicit"]:
                sm, st = models["stage5_sentiment_fully_implicit"]
                preds.fully_implicit_sentiment = classify_span_pair(sm, st, tokens, device, SENTIMENT_ID2LABEL,
                                                                      None, None, max_length)

    return assemble_quads(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test_data", default=None, help="Overrides config's test_data if given.")
    ap.add_argument("--output", default="stage6_results.json")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    test_path = args.test_data or cfg["test_data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device}...")
    models, category_id2label = load_all_models(cfg, device)

    pred_by_sentence, gold_by_sentence = {}, {}
    with open(test_path, encoding="utf-8") as f:
        lines = f.readlines()

    from tqdm import tqdm
    for i, line in enumerate(tqdm(lines, desc="Running pipeline")):
        rec = json.loads(line)
        gold_by_sentence[i] = rec["quads"]
        pred_by_sentence[i] = run_pipeline_on_sentence(rec["tokens"], models, category_id2label, device)

    results = evaluate_quads(pred_by_sentence, gold_by_sentence)
    print("\n=== End-to-end results ===")
    for subset, m in results.items():
        print(f"{subset}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
