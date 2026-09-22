"""
Detailed diagnostic script to analyze span boundary errors and pair coverage asymmetry:
1. Pair Asymmetry: How often the model extracts the Aspect (first span) but misses the Opinion (last span), and vice-versa.
2. Boundary Errors: For multi-word spans, how often the model gets the start token (B) but misses the end boundary (I continuation cut short or overshot).

Usage:
    PYTHONPATH=src/stage_aope_sdrn python src/stage_aope_sdrn/analyze_span_errors.py \
        --checkpoint results/stage_aope_sdrn/afroxlmr_crf_run1/best_model.pt \
        --config configs/stage_aope_sdrn.yaml
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze span boundary errors and pair coverage asymmetry.")
    parser.add_argument("--config", default="configs/stage_aope_sdrn.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--subword_aggregation", choices=["entity_first", "first"], default="entity_first")
    parser.add_argument("--opinion_bias", type=float, default=0.0)
    return parser.parse_args()


def classify_span_boundary_match(gold_span: tuple[int, int], pred_spans: list[tuple[int, int]]) -> str:
    """
    Classifies how predicted spans align with a gold span [s, e):
    - 'exact': [s, e) exactly matches
    - 'start_matched_cut_short': predicted [s, e') where s == gold_s and e' < gold_e
    - 'start_matched_overshot': predicted [s, e') where s == gold_s and e' > gold_e
    - 'end_matched_start_mismatched': predicted [s', e) where s' != gold_s and e' == gold_e
    - 'partial_overlap': overlaps [gold_s, gold_e) but neither boundary matches exactly
    - 'completely_missed': zero overlap with any predicted span
    """
    gs, ge = gold_span
    if gold_span in pred_spans:
        return "exact"

    # Check for start match
    for ps, pe in pred_spans:
        if ps == gs:
            if pe < ge:
                return "start_matched_cut_short"
            elif pe > ge:
                return "start_matched_overshot"

    # Check for end match
    for ps, pe in pred_spans:
        if pe == ge:
            return "end_matched_start_mismatched"

    # Check for partial overlap
    for ps, pe in pred_spans:
        if max(gs, ps) < min(ge, pe):
            return "partial_overlap"

    return "completely_missed"


def main():
    import torch
    import yaml
    from align import decode_subword_predictions_5way
    from bio_labels import decode_5way_bio_spans
    from dataset import JointAOPEDataset, collate_fn
    from model import JointAOPESDRN
    from relation_utils import explicit_pairs
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer

    args = parse_args()

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
            checkpoint = "results/stage_aope_sdrn/afroxlmr_crf_run1/best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint}")
    print(f"Test dataset: {test_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_ds = JointAOPEDataset(test_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    training_cfg = cfg.get("training", {})
    bio_class_weights = training_cfg.get("bio_class_weights")
    span_loss_weight = training_cfg.get("span_loss_weight", 1.0)
    use_crf = cfg.get("use_crf", True)

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
    model.eval()

    # Pair-level counters
    pair_total = 0
    both_found = 0
    aspect_found_opinion_missed = 0
    opinion_found_aspect_missed = 0
    neither_found = 0

    # Boundary counters for all spans and multi-word spans
    asp_boundary_counts = {
        "all": {"exact": 0, "start_matched_cut_short": 0, "start_matched_overshot": 0, "end_matched_start_mismatched": 0, "partial_overlap": 0, "completely_missed": 0},
        "multi_word": {"exact": 0, "start_matched_cut_short": 0, "start_matched_overshot": 0, "end_matched_start_mismatched": 0, "partial_overlap": 0, "completely_missed": 0},
    }
    opn_boundary_counts = {
        "all": {"exact": 0, "start_matched_cut_short": 0, "start_matched_overshot": 0, "end_matched_start_mismatched": 0, "partial_overlap": 0, "completely_missed": 0},
        "multi_word": {"exact": 0, "start_matched_cut_short": 0, "start_matched_overshot": 0, "end_matched_start_mismatched": 0, "partial_overlap": 0, "completely_missed": 0},
    }

    rec_idx = 0
    print("Running evaluation and error analysis...")
    for batch in tqdm(loader, desc="Analyzing"):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        content_mask = batch.get("content_mask")
        if content_mask is not None:
            content_mask = content_mask.to(device)
        else:
            content_mask = attn.bool()

        out = model(input_ids=input_ids, attention_mask=attn, content_mask=content_mask)
        pred_ids = model.decode_tags(
            out["logits"],
            mask=content_mask,
            opinion_emission_bias=args.opinion_bias,
        )

        bsz = input_ids.size(0)
        for i in range(bsz):
            rec = test_ds.records[rec_idx]
            rec_idx += 1
            enc = tokenizer(
                rec["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=test_ds.max_length,
            )
            word_ids = enc.word_ids(batch_index=0)

            word_tags = decode_subword_predictions_5way(
                word_ids,
                pred_ids[i][:len(word_ids)],
                strategy=args.subword_aggregation,
            )
            pred_aspects, pred_opinions = decode_5way_bio_spans(word_tags)
            g_pairs = explicit_pairs(rec["quads"])

            # 1. Pair coverage analysis
            for a_span, o_span in g_pairs:
                pair_total += 1
                a_hit = a_span in pred_aspects
                o_hit = o_span in pred_opinions

                if a_hit and o_hit:
                    both_found += 1
                elif a_hit and not o_hit:
                    aspect_found_opinion_missed += 1
                elif not a_hit and o_hit:
                    opinion_found_aspect_missed += 1
                else:
                    neither_found += 1

            # 2. Boundary analysis for aspects
            g_aspects = [p[0] for p in g_pairs]
            for a_span in g_aspects:
                cat = classify_span_boundary_match(a_span, pred_aspects)
                asp_boundary_counts["all"][cat] += 1
                if (a_span[1] - a_span[0]) > 1:
                    asp_boundary_counts["multi_word"][cat] += 1

            # 3. Boundary analysis for opinions
            g_opinions = [p[1] for p in g_pairs]
            for o_span in g_opinions:
                cat = classify_span_boundary_match(o_span, pred_opinions)
                opn_boundary_counts["all"][cat] += 1
                if (o_span[1] - o_span[0]) > 1:
                    opn_boundary_counts["multi_word"][cat] += 1

    print("\n" + "=" * 80)
    print("1. ASPECT-OPINION PAIR COVERAGE ASYMMETRY (FIRST SPAN VS. LAST SPAN)")
    print("=" * 80)
    print(f"Total Ground-Truth Pairs Evaluated : {pair_total}")
    print(f"  • Both Found (Candidate Ceiling) : {both_found:>5} ({both_found / pair_total * 100:>6.2f}%)")
    print(f"  • Aspect Found, Opinion MISSED   : {aspect_found_opinion_missed:>5} ({aspect_found_opinion_missed / pair_total * 100:>6.2f}%)  <-- [Got 1st span, missed last span]")
    print(f"  • Opinion Found, Aspect MISSED   : {opinion_found_aspect_missed:>5} ({opinion_found_aspect_missed / pair_total * 100:>6.2f}%)")
    print(f"  • Neither Found                  : {neither_found:>5} ({neither_found / pair_total * 100:>6.2f}%)")
    print("-" * 80)
    aspect_hits = both_found + aspect_found_opinion_missed
    if aspect_hits > 0:
        print(f"Conditional: When the model gets the Aspect, it misses the Opinion: {aspect_found_opinion_missed / aspect_hits * 100:.2f}% of the time.")
    lost_pairs = pair_total - both_found
    if lost_pairs > 0:
        print(f"Attribution of Lost Pairs: {aspect_found_opinion_missed / lost_pairs * 100:.2f}% of unformed pairs failed solely due to the missing Opinion.")

    print("\n" + "=" * 80)
    print("2. MULTI-WORD OPINION BOUNDARY ERRORS (START VS. END BOUNDARY)")
    print("=" * 80)
    opn_multi_total = sum(opn_boundary_counts["multi_word"].values())
    print(f"Total Multi-Word Opinions (length >= 2): {opn_multi_total}")
    for k, v in opn_boundary_counts["multi_word"].items():
        pct = (v / opn_multi_total * 100) if opn_multi_total > 0 else 0
        desc = ""
        if k == "start_matched_cut_short":
            desc = " <-- [Got first token/start, missed last token/cut short]"
        elif k == "start_matched_overshot":
            desc = " <-- [Got first token/start, overshot last token]"
        print(f"  • {k:<30s}: {v:>5} ({pct:>6.2f}%){desc}")

    print("\n" + "=" * 80)
    print("3. MULTI-WORD ASPECT BOUNDARY ERRORS (START VS. END BOUNDARY)")
    print("=" * 80)
    asp_multi_total = sum(asp_boundary_counts["multi_word"].values())
    print(f"Total Multi-Word Aspects (length >= 2): {asp_multi_total}")
    for k, v in asp_boundary_counts["multi_word"].items():
        pct = (v / asp_multi_total * 100) if asp_multi_total > 0 else 0
        desc = ""
        if k == "start_matched_cut_short":
            desc = " <-- [Got first token/start, missed last token/cut short]"
        elif k == "start_matched_overshot":
            desc = " <-- [Got first token/start, overshot last token]"
        print(f"  • {k:<30s}: {v:>5} ({pct:>6.2f}%){desc}")
    print("=" * 80)


if __name__ == "__main__":
    main()
