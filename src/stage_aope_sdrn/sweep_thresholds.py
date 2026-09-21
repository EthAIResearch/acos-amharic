"""
Threshold sweep evaluation for the joint SDRN-style AOPE model.

Evaluates an existing checkpoint across a range of correlation degree thresholds
(delta-hat) in a single inference pass, reporting:
  1. Aspect and opinion span extraction PRF (the recall bottleneck)
  2. The candidate pair recall ceiling (upper bound if all candidate pairs were accepted)
  3. A threshold sweep table for pair Precision, Recall, and F1

Usage:
    python sweep_thresholds.py --checkpoint results/stage_aope_sdrn/roberta_base_run1/best_model.pt \
                               --config configs/stage_aope_sdrn.yaml
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

import torch
import yaml
from align import decode_subword_predictions, decode_subword_predictions_5way
from bio_labels import decode_5way_bio_spans, decode_bio_spans
from dataset import JointAOPEDataset, collate_fn
from model import JointAOPESDRN
from relation_utils import (
    compute_prf,
    correlation_degree,
    explicit_pairs,
    sweep_thresholds,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep relation thresholds on SDRN checkpoint.")
    parser.add_argument("--config", default="configs/stage_aope_sdrn.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint best_model.pt. If omitted, derived from config output_dir.")
    parser.add_argument("--test", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--use_crf", action="store_true", default=None,
                        help="Use Linear-Chain CRF decoding. Defaults to config setting.")
    parser.add_argument("--no_crf", dest="use_crf", action="store_false")
    return parser.parse_args()


@torch.no_grad()
def run_sweep(model, dataset, tokenizer, device, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_pred_aspects, all_gold_aspects = [], []
    all_pred_opinions, all_gold_opinions = [], []
    candidates_with_scores_per_ex = []
    all_gold_pairs = []
    rec_idx = 0

    print("Running forward inference pass over test set...")
    for batch in tqdm(loader, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        content_mask = batch.get("content_mask")
        if content_mask is not None:
            content_mask = content_mask.to(device)
        else:
            content_mask = attn.bool()

        out = model(input_ids=input_ids, attention_mask=attn, content_mask=content_mask)
        pred_ids = model.decode_tags(out["logits"], mask=content_mask)
        rel_scores = out["relation_logits"].cpu().tolist()

        bsz = input_ids.size(0)
        for i in range(bsz):
            rec = dataset.records[rec_idx]
            rec_idx += 1
            enc = tokenizer(
                rec["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=dataset.max_length,
            )
            word_ids = enc.word_ids(batch_index=0)

            word_tags = decode_subword_predictions_5way(word_ids, pred_ids[i][:len(word_ids)])
            a_spans, o_spans = decode_5way_bio_spans(word_tags)

            g_pairs = explicit_pairs(rec["quads"])
            g_aspects = [p[0] for p in g_pairs]
            g_opinions = [p[1] for p in g_pairs]

            all_pred_aspects.append(a_spans)
            all_gold_aspects.append(g_aspects)
            all_pred_opinions.append(o_spans)
            all_gold_opinions.append(g_opinions)
            all_gold_pairs.append(g_pairs)

            # Map subword-level relation matrix back to words via max-pooling
            # across all subwords for words wi and wj
            n_words = len(rec["tokens"])
            word_to_subwords = {w: [] for w in range(n_words)}
            for si, wid in enumerate(word_ids):
                if wid is not None and wid < n_words:
                    word_to_subwords[wid].append(si)

            word_rel = [[0.0] * n_words for _ in range(n_words)]
            for wi in range(n_words):
                for wj in range(n_words):
                    sis = word_to_subwords[wi]
                    sjs = word_to_subwords[wj]
                    if sis and sjs:
                        sub_scores = [
                            rel_scores[i][si][sj]
                            for si in sis
                            for sj in sjs
                            if si < len(rel_scores[i]) and sj < len(rel_scores[i])
                        ]
                        if sub_scores:
                            word_rel[wi][wj] = max(sub_scores)

            # Cache scored candidate pairs
            ex_candidates = []
            for a_span in a_spans:
                for o_span in o_spans:
                    score = correlation_degree(word_rel, a_span, o_span)
                    ex_candidates.append(((a_span, o_span), score))
            candidates_with_scores_per_ex.append(ex_candidates)

    aspect_prf = compute_prf(all_pred_aspects, all_gold_aspects)
    opinion_prf = compute_prf(all_pred_opinions, all_gold_opinions)
    sweep_data = sweep_thresholds(candidates_with_scores_per_ex, all_gold_pairs)

    return {
        "aspect_metrics": aspect_prf,
        "opinion_metrics": opinion_prf,
        "theoretical_pair_recall_ceiling": aspect_prf["recall"] * opinion_prf["recall"],
        "sweep_results": sweep_data,
    }


def main():
    args = parse_args()

    # Load config defaults
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg.get("model_name", "Davlan/afro-xlmr-base")
    test_path = args.test or cfg.get("data", {}).get("test", "data/prepared_explicit/test.jsonl")
    num_recurrent_steps = cfg.get("num_recurrent_steps", 2)
    relation_threshold = cfg.get("relation_threshold", 0.1)

    checkpoint = args.checkpoint
    if not checkpoint:
        cfg_out = cfg.get("output_dir")
        if cfg_out:
            checkpoint = os.path.join(cfg_out, "best_model.pt")
        else:
            checkpoint = "results/stage_aope_sdrn/afroxlmr_base_run1/best_model.pt"

    use_crf = args.use_crf if args.use_crf is not None else cfg.get("use_crf", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint}")
    print(f"Test dataset: {test_path}")
    print(f"Use CRF decoding: {use_crf}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_ds = JointAOPEDataset(test_path, tokenizer, max_length=args.max_length)

    training_cfg = cfg.get("training", {})
    bio_class_weights = training_cfg.get("bio_class_weights")
    span_loss_weight = training_cfg.get("span_loss_weight", 1.0)

    model = JointAOPESDRN(
        model_name,
        num_recurrent_steps=num_recurrent_steps,
        relation_threshold=relation_threshold,
        bio_class_weights=bio_class_weights,
        span_loss_weight=span_loss_weight,
        use_crf=use_crf,
    )
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    results = run_sweep(model, test_ds, tokenizer, device, batch_size=args.batch_size)

    # Output report
    a_m = results["aspect_metrics"]
    o_m = results["opinion_metrics"]
    s_d = results["sweep_results"]

    print("\n" + "=" * 78)
    print("SPAN EXTRACTION METRICS (ROOT BOTTLENECK ANALYSIS)")
    print("=" * 78)
    print(f"Aspect Spans  : Precision={a_m['precision']:.4f}  Recall={a_m['recall']:.4f}  F1={a_m['f1']:.4f}  (TP={a_m['tp']}, FP={a_m['fp']}, FN={a_m['fn']})")
    print(f"Opinion Spans : Precision={o_m['precision']:.4f}  Recall={o_m['recall']:.4f}  F1={o_m['f1']:.4f}  (TP={o_m['tp']}, FP={o_m['fp']}, FN={o_m['fn']})")
    print(f"Theoretical Pair Recall Ceiling (Aspect R * Opinion R) : {results['theoretical_pair_recall_ceiling'] * 100:.2f}%")
    print(f"Candidate Pair Recall Ceiling (All proposed pairs)     : {s_d['candidate_ceiling_recall'] * 100:.2f}%")

    print("\n" + "=" * 78)
    print("THRESHOLD SWEEP TABLE (delta-hat vs. Pair Precision, Recall, F1)")
    print("=" * 78)
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1 Score':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6}")
    print("-" * 78)
    for row in s_d["sweep"]:
        is_best = " *" if row["threshold"] == round(s_d["best_threshold"], 4) else ""
        print(f"{row['threshold']:>10.2f} | {row['precision']:>10.4f} | {row['recall']:>10.4f} | {row['f1']:>10.4f}{is_best} | {row['tp']:>6} | {row['fp']:>6} | {row['fn']:>6}")

    print("=" * 78)
    b_m = s_d["best_metrics"]
    print(f"Optimal Threshold by F1: delta-hat = {s_d['best_threshold']:.2f}")
    print(f"Optimal Pair Metrics   : Precision={b_m['precision']:.4f}, Recall={b_m['recall']:.4f}, F1={b_m['f1']:.4f}")

    # Save results
    out_dir = os.path.dirname(checkpoint) if checkpoint and os.path.dirname(checkpoint) else "."
    out_file = args.output_file or os.path.join(out_dir, "threshold_sweep.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full sweep results to: {out_file}")


if __name__ == "__main__":
    main()
