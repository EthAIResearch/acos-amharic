"""
Seen vs. Unseen Span Recall Diagnostic
=======================================
Breaks down model recall into *seen* (appeared in training) vs *unseen* (novel)
opinion and aspect spans at three granularity levels:

  1. **Exact span** — the full multi-word surface form appeared in training
  2. **Token overlap** — at least one token of the span appeared in some
     training opinion/aspect span
  3. **Character trigram overlap** — the span shares ≥50% of its character
     trigrams with at least one training span (proxy for morphological root/
     stem overlap without a dedicated Amharic morphological analyzer)

This directly answers: "Is the opinion recall gap driven by morphological
variants of seen words (fixable with byte-level/continued pretraining) or
truly novel vocabulary (data coverage problem)?"

Usage:
    PYTHONPATH=src/stage_aope_sdrn python src/stage_aope_sdrn/diagnose_seen_unseen.py \
        --checkpoint results/stage_aope_sdrn/afroxlmr_crf_run1/best_model.pt

    # Pure vocabulary analysis without model inference (no checkpoint needed):
    PYTHONPATH=src/stage_aope_sdrn python src/stage_aope_sdrn/diagnose_seen_unseen.py \
        --vocab-only
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seen vs. Unseen span recall diagnostic."
    )
    parser.add_argument("--config", default="configs/stage_aope_sdrn.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train", default=None, help="Training JSONL path")
    parser.add_argument("--test", default=None, help="Test JSONL path")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--subword_aggregation",
        choices=["entity_first", "first"],
        default="entity_first",
    )
    parser.add_argument("--opinion_bias", type=float, default=0.0)
    parser.add_argument(
        "--trigram_threshold",
        type=float,
        default=0.50,
        help="Min fraction of shared trigrams for 'trigram-seen' (default: 0.50)",
    )
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="Only report vocabulary overlap statistics (no model inference)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_spans(records: list[dict], span_type: str) -> list[tuple[str, list[str]]]:
    """
    Extract (surface_text, token_list) for all aspect or opinion spans.
    span_type: 'aspect' or 'opinion'
    """
    from relation_utils import explicit_pairs

    spans = []
    for rec in records:
        tokens = rec["tokens"]
        pairs = explicit_pairs(rec["quads"])
        for a_span, o_span in pairs:
            if span_type == "aspect":
                s, e = a_span
            else:
                s, e = o_span
            span_tokens = tokens[s:e]
            surface = " ".join(span_tokens)
            spans.append((surface, span_tokens))
    return spans


def char_trigrams(text: str) -> set[str]:
    """Extract character trigrams from a string."""
    if len(text) < 3:
        return {text}
    return {text[i : i + 3] for i in range(len(text) - 2)}


def build_vocabulary(
    train_spans: list[tuple[str, list[str]]],
) -> dict:
    """
    Build training vocabulary at three granularities:
      - exact_spans: set of full surface forms
      - tokens: set of individual tokens
      - trigram_index: dict mapping each trigram -> set of training span surfaces
    """
    exact_spans = set()
    token_set = set()
    trigram_sets = {}  # surface -> set of trigrams (for Jaccard later)

    for surface, tok_list in train_spans:
        exact_spans.add(surface)
        for t in tok_list:
            token_set.add(t)
        trigram_sets[surface] = char_trigrams(surface)

    return {
        "exact_spans": exact_spans,
        "tokens": token_set,
        "trigram_sets": trigram_sets,
    }


def classify_span(
    surface: str,
    tok_list: list[str],
    vocab: dict,
    trigram_threshold: float = 0.50,
) -> dict:
    """
    Classify a test span against the training vocabulary.
    Returns dict with boolean keys for each granularity.
    """
    exact_seen = surface in vocab["exact_spans"]
    token_seen = any(t in vocab["tokens"] for t in tok_list)

    # Trigram overlap: max Jaccard similarity with any training span
    test_tg = char_trigrams(surface)
    max_jaccard = 0.0
    best_match = None
    if test_tg:
        for train_surface, train_tg in vocab["trigram_sets"].items():
            if not train_tg:
                continue
            intersection = len(test_tg & train_tg)
            union = len(test_tg | train_tg)
            jacc = intersection / union if union > 0 else 0.0
            if jacc > max_jaccard:
                max_jaccard = jacc
                best_match = train_surface
    trigram_seen = max_jaccard >= trigram_threshold

    return {
        "exact_seen": exact_seen,
        "token_seen": token_seen,
        "trigram_seen": trigram_seen,
        "trigram_jaccard": max_jaccard,
        "trigram_best_match": best_match,
    }


# ---------------------------------------------------------------------------
# Vocabulary-only analysis
# ---------------------------------------------------------------------------

def vocab_only_analysis(
    train_path: str,
    test_path: str,
    trigram_threshold: float,
) -> dict:
    """Report vocabulary overlap statistics without model inference."""
    train_records = load_records(train_path)
    test_records = load_records(test_path)

    results = {}
    for span_type in ("opinion", "aspect"):
        train_spans = extract_spans(train_records, span_type)
        test_spans = extract_spans(test_records, span_type)
        vocab = build_vocabulary(train_spans)

        counts = {
            "total": len(test_spans),
            "unique_train_spans": len(vocab["exact_spans"]),
            "unique_train_tokens": len(vocab["tokens"]),
            "exact_seen": 0,
            "exact_unseen": 0,
            "token_seen": 0,
            "token_unseen": 0,
            "trigram_seen": 0,
            "trigram_unseen": 0,
            "multi_word": 0,
            "single_word": 0,
            "multi_word_exact_unseen": 0,
            "single_word_exact_unseen": 0,
        }

        # Track unique unseen surfaces for reporting
        unseen_examples = defaultdict(int)
        unseen_no_token_overlap = defaultdict(int)
        unseen_no_trigram_overlap = []

        for surface, tok_list in test_spans:
            cls = classify_span(surface, tok_list, vocab, trigram_threshold)
            is_multi = len(tok_list) > 1

            if cls["exact_seen"]:
                counts["exact_seen"] += 1
            else:
                counts["exact_unseen"] += 1
                unseen_examples[surface] += 1

            if cls["token_seen"]:
                counts["token_seen"] += 1
            else:
                counts["token_unseen"] += 1
                unseen_no_token_overlap[surface] += 1

            if cls["trigram_seen"]:
                counts["trigram_seen"] += 1
            else:
                counts["trigram_unseen"] += 1
                if not cls["exact_seen"]:
                    unseen_no_trigram_overlap.append(
                        (surface, cls["trigram_jaccard"], cls["trigram_best_match"])
                    )

            if is_multi:
                counts["multi_word"] += 1
                if not cls["exact_seen"]:
                    counts["multi_word_exact_unseen"] += 1
            else:
                counts["single_word"] += 1
                if not cls["exact_seen"]:
                    counts["single_word_exact_unseen"] += 1

        # Top unseen examples
        top_unseen = sorted(unseen_examples.items(), key=lambda x: -x[1])[:20]
        top_truly_novel = sorted(
            unseen_no_token_overlap.items(), key=lambda x: -x[1]
        )[:20]

        counts["top_unseen_spans"] = top_unseen
        counts["top_truly_novel_spans"] = top_truly_novel
        counts["example_no_trigram_overlap"] = unseen_no_trigram_overlap[:10]
        results[span_type] = counts

    return results


def print_vocab_report(results: dict):
    """Pretty-print vocabulary overlap analysis."""
    for span_type in ("opinion", "aspect"):
        c = results[span_type]
        total = c["total"]
        print(f"\n{'=' * 80}")
        print(f"  {span_type.upper()} VOCABULARY OVERLAP (Train → Test)")
        print(f"{'=' * 80}")
        print(f"  Train unique spans : {c['unique_train_spans']:,}")
        print(f"  Train unique tokens: {c['unique_train_tokens']:,}")
        print(f"  Test span instances : {total:,}")
        print()

        def pct(n: int, t: int = total) -> str:
            return f"{n / t * 100:.1f}%" if t > 0 else "N/A"

        print("  Granularity          │ Seen          │ Unseen")
        print("  ─────────────────────┼───────────────┼───────────────")
        print(
            f"  Exact span match     │ {c['exact_seen']:>5} ({pct(c['exact_seen']):>5}) │ "
            f"{c['exact_unseen']:>5} ({pct(c['exact_unseen']):>5})"
        )
        print(
            f"  ≥1 token overlap     │ {c['token_seen']:>5} ({pct(c['token_seen']):>5}) │ "
            f"{c['token_unseen']:>5} ({pct(c['token_unseen']):>5})"
        )
        print(
            f"  Trigram overlap ≥50% │ {c['trigram_seen']:>5} ({pct(c['trigram_seen']):>5}) │ "
            f"{c['trigram_unseen']:>5} ({pct(c['trigram_unseen']):>5})"
        )
        print()

        print("  Word-length breakdown of exact-unseen:")
        print(
            f"    Single-word unseen: {c['single_word_exact_unseen']} / {c['single_word']} "
            f"({c['single_word_exact_unseen'] / c['single_word'] * 100:.1f}%)"
            if c["single_word"] > 0
            else "    Single-word: N/A"
        )
        print(
            f"    Multi-word unseen : {c['multi_word_exact_unseen']} / {c['multi_word']} "
            f"({c['multi_word_exact_unseen'] / c['multi_word'] * 100:.1f}%)"
            if c["multi_word"] > 0
            else "    Multi-word: N/A"
        )
        print()

        if c["top_unseen_spans"]:
            print(f"  Top 20 most frequent exact-unseen {span_type} spans:")
            for surf, cnt in c["top_unseen_spans"]:
                print(f"    {cnt:>3}x  {surf}")
            print()

        if c["top_truly_novel_spans"]:
            print(
                f"  Top 20 truly novel {span_type} spans (no token overlap with train):"
            )
            for surf, cnt in c["top_truly_novel_spans"]:
                print(f"    {cnt:>3}x  {surf}")
            print()

        if c["example_no_trigram_overlap"]:
            print(
                "  Examples with NO trigram overlap (completely novel morphology):"
            )
            for surf, jacc, best in c["example_no_trigram_overlap"]:
                best_str = best if best else "—"
                print(f"    {surf}  (max Jaccard={jacc:.3f}, closest={best_str})")


# ---------------------------------------------------------------------------
# Model-based seen/unseen recall breakdown
# ---------------------------------------------------------------------------

def model_recall_analysis(args, train_path: str, test_path: str, trigram_threshold: float) -> dict:
    """Run model inference and report recall bucketed by seen/unseen."""
    import torch
    import yaml
    from align import ID2LABEL_5WAY, decode_subword_predictions_5way
    from bio_labels import decode_5way_bio_spans
    from dataset import JointAOPEDataset, collate_fn
    from model import JointAOPESDRN
    from relation_utils import explicit_pairs
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer

    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg.get("model_name", "Davlan/afro-xlmr-base")

    # Build training vocabulary
    print("Building training vocabulary...")
    train_records = load_records(train_path)
    train_opn_spans = extract_spans(train_records, "opinion")
    train_asp_spans = extract_spans(train_records, "aspect")
    opn_vocab = build_vocabulary(train_opn_spans)
    asp_vocab = build_vocabulary(train_asp_spans)
    print(
        f"  Opinion vocab: {len(opn_vocab['exact_spans']):,} unique spans, "
        f"{len(opn_vocab['tokens']):,} unique tokens"
    )
    print(
        f"  Aspect vocab : {len(asp_vocab['exact_spans']):,} unique spans, "
        f"{len(asp_vocab['tokens']):,} unique tokens"
    )

    # Load model
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_ds = JointAOPEDataset(test_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    training_cfg = cfg.get("training", {})
    bio_class_weights = training_cfg.get("bio_class_weights")
    span_loss_weight = training_cfg.get("span_loss_weight", 1.0)
    use_crf = cfg.get("use_crf", True)

    model = JointAOPESDRN(
        model_name,
        num_recurrent_steps=cfg.get("num_recurrent_steps", 2),
        relation_threshold=cfg.get("relation_threshold", 0.1),
        bio_class_weights=bio_class_weights,
        span_loss_weight=span_loss_weight,
        use_crf=use_crf,
    )
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Per-bucket counters: {(span_type, granularity, bucket) -> {tp, fn}}
    counters = defaultdict(lambda: {"tp": 0, "fn": 0, "total": 0})
    # Track per-example details for interesting cases
    missed_seen_opinions = []  # Seen opinions the model missed (diagnostic gold)
    found_unseen_opinions = []  # Unseen opinions the model found (model strength)

    rec_idx = 0
    print("Running model inference...")
    for batch in tqdm(loader, desc="Diagnosing"):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        content_mask = batch.get("content_mask")
        if content_mask is not None:
            content_mask = content_mask.to(device)
        else:
            content_mask = attn.bool()

        subword_to_word = batch.get("subword_to_word")
        word_mask = batch.get("word_mask")
        if subword_to_word is not None:
            subword_to_word = subword_to_word.to(device)
            word_mask = word_mask.to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                content_mask=content_mask,
                subword_to_word=subword_to_word,
                word_mask=word_mask,
            )
        is_word_level = out.get("is_word_level", False)
        decode_mask = word_mask if is_word_level else content_mask
        pred_ids = model.decode_tags(
            out["logits"],
            mask=decode_mask,
            opinion_emission_bias=args.opinion_bias,
        )

        bsz = input_ids.size(0)
        for i in range(bsz):
            rec = test_ds.records[rec_idx]
            rec_idx += 1
            tokens = rec["tokens"]
            n_words = len(tokens)

            if is_word_level:
                word_tags = [
                    ID2LABEL_5WAY.get(pid, "O") for pid in pred_ids[i][:n_words]
                ]
            else:
                enc = tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=test_ds.max_length,
                )
                word_ids = enc.word_ids(batch_index=0)
                word_tags = decode_subword_predictions_5way(
                    word_ids,
                    pred_ids[i][: len(word_ids)],
                    strategy=args.subword_aggregation,
                )

            pred_aspects, pred_opinions = decode_5way_bio_spans(word_tags)
            pairs = explicit_pairs(rec["quads"])

            for a_span, o_span in pairs:
                # Classify and score opinion
                o_surface = " ".join(tokens[o_span[0] : o_span[1]])
                o_toks = tokens[o_span[0] : o_span[1]]
                o_cls = classify_span(o_surface, o_toks, opn_vocab, trigram_threshold)
                o_hit = o_span in pred_opinions

                for gran, key in [
                    ("exact", "exact_seen"),
                    ("token", "token_seen"),
                    ("trigram", "trigram_seen"),
                ]:
                    bucket = "seen" if o_cls[key] else "unseen"
                    c = counters[("opinion", gran, bucket)]
                    c["total"] += 1
                    if o_hit:
                        c["tp"] += 1
                    else:
                        c["fn"] += 1

                # Also track word length
                is_multi = len(o_toks) > 1
                wlen_key = "multi_word" if is_multi else "single_word"
                c = counters[("opinion", "word_length", wlen_key)]
                c["total"] += 1
                if o_hit:
                    c["tp"] += 1
                else:
                    c["fn"] += 1

                # Cross-analysis: word_length × exact_seen
                exact_bucket = "seen" if o_cls["exact_seen"] else "unseen"
                cross_key = f"{wlen_key}_{exact_bucket}"
                c = counters[("opinion", "cross", cross_key)]
                c["total"] += 1
                if o_hit:
                    c["tp"] += 1
                else:
                    c["fn"] += 1

                # Collect examples for qualitative analysis
                if not o_hit and o_cls["exact_seen"] and len(missed_seen_opinions) < 50:
                    missed_seen_opinions.append(
                        {
                            "surface": o_surface,
                            "sentence": " ".join(tokens[:40]),
                            "span": list(o_span),
                        }
                    )
                if o_hit and not o_cls["exact_seen"] and len(found_unseen_opinions) < 50:
                    found_unseen_opinions.append(
                        {
                            "surface": o_surface,
                            "jaccard": round(o_cls["trigram_jaccard"], 3),
                            "closest_train": o_cls["trigram_best_match"],
                        }
                    )

                # Classify and score aspect (control comparison)
                a_surface = " ".join(tokens[a_span[0] : a_span[1]])
                a_toks = tokens[a_span[0] : a_span[1]]
                a_cls = classify_span(a_surface, a_toks, asp_vocab, trigram_threshold)
                a_hit = a_span in pred_aspects

                for gran, key in [
                    ("exact", "exact_seen"),
                    ("token", "token_seen"),
                    ("trigram", "trigram_seen"),
                ]:
                    bucket = "seen" if a_cls[key] else "unseen"
                    c = counters[("aspect", gran, bucket)]
                    c["total"] += 1
                    if a_hit:
                        c["tp"] += 1
                    else:
                        c["fn"] += 1

    return {
        "counters": dict(counters),
        "missed_seen_opinions": missed_seen_opinions,
        "found_unseen_opinions": found_unseen_opinions,
    }


def print_model_report(results: dict):
    """Pretty-print model-based seen/unseen recall breakdown."""
    counters = results["counters"]

    for span_type in ("opinion", "aspect"):
        print(f"\n{'=' * 80}")
        print(f"  {span_type.upper()} RECALL: SEEN vs UNSEEN BREAKDOWN")
        print(f"{'=' * 80}")

        def recall_str(key):
            c = counters.get(key, {"tp": 0, "fn": 0, "total": 0})
            total = c["total"]
            if total == 0:
                return "   — / —      (—)"
            r = c["tp"] / total
            return f"{c['tp']:>5} / {total:<5} ({r * 100:>5.1f}%)"

        print("\n  By vocabulary granularity:")
        print(f"  {'Granularity':<25s} │ {'Seen Recall':<25s} │ {'Unseen Recall':<25s}")
        print(f"  {'─' * 25}┼{'─' * 27}┼{'─' * 27}")
        for gran in ("exact", "token", "trigram"):
            s = recall_str((span_type, gran, "seen"))
            u = recall_str((span_type, gran, "unseen"))
            label = {
                "exact": "Exact span match",
                "token": "≥1 token overlap",
                "trigram": "Trigram Jaccard ≥50%",
            }[gran]
            print(f"  {label:<25s} │ {s:<25s} │ {u:<25s}")

        if span_type == "opinion":
            print("\n  By word length:")
            print(f"  {'Length':<25s} │ {'Recall':<25s}")
            print(f"  {'─' * 25}┼{'─' * 27}")
            for wl in ("single_word", "multi_word"):
                s = recall_str(("opinion", "word_length", wl))
                label = "Single-word" if wl == "single_word" else "Multi-word"
                print(f"  {label:<25s} │ {s:<25s}")

            print("\n  Cross-analysis (word length × exact-seen):")
            print(
                f"  {'Bucket':<30s} │ {'Recall':<25s}"
            )
            print(f"  {'─' * 30}┼{'─' * 27}")
            for cross in (
                "single_word_seen",
                "single_word_unseen",
                "multi_word_seen",
                "multi_word_unseen",
            ):
                s = recall_str(("opinion", "cross", cross))
                label = cross.replace("_", " ").title()
                print(f"  {label:<30s} │ {s:<25s}")

    # Qualitative examples
    if results["missed_seen_opinions"]:
        print(f"\n{'=' * 80}")
        print("  MISSED SEEN OPINIONS (model fails despite training exposure)")
        print(f"{'=' * 80}")
        for ex in results["missed_seen_opinions"][:15]:
            print(f"  Opinion: '{ex['surface']}'")
            print(f"    Sentence: {ex['sentence']}...")
            print()

    if results["found_unseen_opinions"]:
        print(f"\n{'=' * 80}")
        print("  FOUND UNSEEN OPINIONS (model generalizes beyond training)")
        print(f"{'=' * 80}")
        for ex in results["found_unseen_opinions"][:15]:
            closest = ex["closest_train"] or "—"
            print(
                f"  Opinion: '{ex['surface']}'  "
                f"(Jaccard={ex['jaccard']:.3f}, closest='{closest}')"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f) or {}

    train_path = args.train or cfg.get("data", {}).get(
        "train", "data/prepared_explicit/train.jsonl"
    )
    test_path = args.test or cfg.get("data", {}).get(
        "test", "data/prepared_explicit/test.jsonl"
    )

    print(f"Train: {train_path}")
    print(f"Test : {test_path}")
    print(f"Trigram threshold: {args.trigram_threshold}")

    # Always run vocabulary analysis
    print("\n" + "━" * 80)
    print("  SECTION 1: VOCABULARY OVERLAP ANALYSIS (no model needed)")
    print("━" * 80)
    vocab_results = vocab_only_analysis(train_path, test_path, args.trigram_threshold)
    print_vocab_report(vocab_results)

    if args.vocab_only:
        print("\n[--vocab-only mode, skipping model inference]")
    else:
        print("\n" + "━" * 80)
        print("  SECTION 2: MODEL RECALL × SEEN/UNSEEN BREAKDOWN")
        print("━" * 80)
        model_results = model_recall_analysis(
            args, train_path, test_path, args.trigram_threshold
        )
        print_model_report(model_results)

        if args.output:
            # Serialize results to JSON
            serializable = {
                "vocab": vocab_results,
                "model": {
                    "counters": {
                        str(k): v for k, v in model_results["counters"].items()
                    },
                    "missed_seen_opinions": model_results["missed_seen_opinions"],
                    "found_unseen_opinions": model_results["found_unseen_opinions"],
                },
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
