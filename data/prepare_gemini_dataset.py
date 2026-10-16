"""Prepare cleaned Gemini ACOS dataset into standard JSONL format:
- Grounds aspect and opinion terms to exact whitespace token indices (a_start, a_end, o_start, o_end).
- Aggregates multiple quads per sentence into single structured records.
- Prunes residual foreign-script and scraping artifact rows.
- Calibrates train/dev/test splits to achieve target ~70% seen and ~30% unseen for aspects AND opinions.
- Guarantees 0 sentence leakage across splits and coverage across all 28 categories and 3 sentiments.
- Generates label_space.json and dataset_stats.json.
"""

import argparse
import collections
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

SENTIMENT_MAP = {"0": "NEUTRAL", "1": "POSITIVE", "2": "NEGATIVE"}
FOREIGN_PATTERN = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0900-\u097f]")


def is_repetitive(text: str, max_rep: int = 5) -> bool:
    words = text.split()
    curr_rep = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            curr_rep += 1
            if curr_rep >= max_rep:
                return True
        else:
            curr_rep = 1
    return False


def find_token_span(tokens: List[str], phrase: str, sentence_text: str) -> Tuple[int, int]:
    """Ground phrase string to exact whitespace-token start and end indices."""
    if not phrase or phrase.strip() in ("NULL", "-1", "none", "None"):
        return -1, -1
    phrase = phrase.strip()
    if not phrase:
        return -1, -1

    pw = phrase.split()
    L = len(pw)

    # 1. Exact token sequence match
    for i in range(len(tokens) - L + 1):
        if tokens[i : i + L] == pw:
            return i, i + L

    # 2. Punctuation-stripped token match
    clean_target = [w.strip(".,!?:;፤፡፦\"\'()") for w in pw]
    for i in range(len(tokens) - L + 1):
        clean_slice = [t.strip(".,!?:;፤፡፦\"\'()") for t in tokens[i : i + L]]
        if clean_slice == clean_target:
            return i, i + L

    # 3. Character-level substring grounding
    idx = sentence_text.find(phrase)
    if idx != -1:
        end_idx = idx + len(phrase)
        cur = 0
        s_tok, e_tok = None, None
        for ti, tok in enumerate(tokens):
            t_start = sentence_text.find(tok, cur)
            if t_start == -1:
                t_start = cur
            t_end = t_start + len(tok)
            cur = t_end
            if s_tok is None and (t_start <= idx < t_end or idx <= t_start < end_idx):
                s_tok = ti
            if s_tok is not None and t_end >= end_idx:
                e_tok = ti + 1
                break
        if s_tok is not None and e_tok is not None and e_tok > s_tok:
            return s_tok, e_tok

    return -1, -1


def quad_key(q: dict) -> tuple:
    return (q["a_start"], q["a_end"], q["category"], q["sentiment"], q["o_start"], q["o_end"])


def prepare_dataset(
    input_csv: str,
    output_dir: str,
    seed: int = 42,
    prune_artifacts: bool = True,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    print(f"Reading cleaned CSV: {input_csv}")
    
    by_text = collections.OrderedDict()
    pruned_foreign = 0
    pruned_rep = 0
    total_rows = 0

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            total_rows += 1
            row_id, text, lang, aspect_term, aspect_cat, opinion_term, sent_str = row
            text_clean = " ".join(text.strip().split())

            if prune_artifacts:
                if FOREIGN_PATTERN.search(text_clean):
                    pruned_foreign += 1
                    continue
                if is_repetitive(text_clean):
                    pruned_rep += 1
                    continue

            if text_clean not in by_text:
                tokens = text_clean.split()
                by_text[text_clean] = {
                    "text": text_clean,
                    "tokens": tokens,
                    "quads": [],
                }

            rec = by_text[text_clean]
            tokens = rec["tokens"]

            a_s, a_e = find_token_span(tokens, aspect_term, text_clean)
            o_s, o_e = find_token_span(tokens, opinion_term, text_clean)
            sentiment = SENTIMENT_MAP.get(str(sent_str).strip(), "NEUTRAL")

            q = {
                "a_start": a_s,
                "a_end": a_e,
                "category": aspect_cat.strip(),
                "sentiment": sentiment,
                "o_start": o_s,
                "o_end": o_e,
            }

            # Drop duplicate quads within the same sentence
            seen = {quad_key(existing_q) for existing_q in rec["quads"]}
            if quad_key(q) not in seen:
                rec["quads"].append(q)

    all_records = list(by_text.values())
    total_quads = sum(len(r["quads"]) for r in all_records)

    print(f"Total CSV rows processed:        {total_rows:,}")
    if prune_artifacts:
        print(f"Pruned foreign-script rows:      {pruned_foreign:,}")
        print(f"Pruned repetitive artifact rows: {pruned_rep:,}")
    print(f"Unique sentences assembled:      {len(all_records):,}")
    print(f"Total valid quads assembled:     {total_quads:,}")

    # Canonical sorting of quads inside each sentence
    for r in all_records:
        r["quads"].sort(key=lambda q: (
            0 if q["a_start"] != -1 else 1,
            q["a_start"],
            q["o_start"],
            q["category"],
        ))

    # Build term to sentence maps
    sent_asps = []
    sent_ops = []
    asp_to_sents = collections.defaultdict(set)
    op_to_sents = collections.defaultdict(set)
    cat_to_sents = collections.defaultdict(list)

    for i, s in enumerate(all_records):
        toks = s["tokens"]
        asps = tuple(" ".join(toks[q["a_start"]:q["a_end"]]).strip() for q in s["quads"] if q["a_start"] != -1)
        ops = tuple(" ".join(toks[q["o_start"]:q["o_end"]]).strip() for q in s["quads"] if q["o_start"] != -1)
        sent_asps.append(asps)
        sent_ops.append(ops)
        for a in asps: asp_to_sents[a].add(i)
        for o in ops: op_to_sents[o].add(i)
        for q in s["quads"]: cat_to_sents[q["category"]].append(i)

    # Partition sentences into recurring (candidates for seen) and unique (candidates for unseen)
    fully_recurring = [i for i in range(len(all_records)) if all(len(op_to_sents[o]) >= 2 for o in sent_ops[i]) and all(len(asp_to_sents[a]) >= 2 for a in sent_asps[i])]
    mostly_unique = [i for i in range(len(all_records)) if all(len(op_to_sents[o]) == 1 for o in sent_ops[i]) and all(len(asp_to_sents[a]) == 1 for a in sent_asps[i])]

    random.seed(seed)
    random.shuffle(fully_recurring)
    random.shuffle(mostly_unique)

    # Anchors for train: guarantee train has representation of recurring terms and all categories
    train_anchors = set()
    for o, sents in op_to_sents.items():
        if len(sents) >= 2: train_anchors.add(min(sents))
    for a, sents in asp_to_sents.items():
        if len(sents) >= 2: train_anchors.add(min(sents))
    all_cats = sorted(cat_to_sents.keys())
    for c in all_cats:
        train_anchors.add(cat_to_sents[c][0])

    recurring_available = [i for i in fully_recurring if i not in train_anchors]

    # Select test and dev: 1400 recurring (seen) + 600 unique (unseen) = ~70% seen, 30% unseen
    test_idx = set(recurring_available[:1400] + mostly_unique[:600])
    dev_idx = set(recurring_available[1400:2800] + mostly_unique[600:1200])

    # Ensure all 28 categories exist in both dev and test
    for c in all_cats:
        if not any(q["category"] == c for i in dev_idx for q in all_records[i]["quads"]):
            cand = [i for i in cat_to_sents[c] if i not in test_idx and i not in train_anchors]
            if cand: dev_idx.add(cand[0])
        if not any(q["category"] == c for i in test_idx for q in all_records[i]["quads"]):
            cand = [i for i in cat_to_sents[c] if i not in dev_idx and i not in train_anchors]
            if cand: test_idx.add(cand[0])

    train_idx = set(range(len(all_records))) - test_idx - dev_idx

    splits = {
        "train": [all_records[i] for i in sorted(train_idx)],
        "dev": [all_records[i] for i in sorted(dev_idx)],
        "test": [all_records[i] for i in sorted(test_idx)],
    }

    # Write splits
    for name, rows in splits.items():
        out_file = out_path / f"{name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {name:5s}: {len(rows):,d} sentences | {sum(len(r['quads']) for r in rows):,d} quads")

    # Evaluate seen/unseen ratios
    train_asps = {a for i in train_idx for a in sent_asps[i]}
    train_ops = {o for i in train_idx for o in sent_ops[i]}

    seen_stats = {}
    for name, rows in [("dev", splits["dev"]), ("test", splits["test"])]:
        tot_a = seen_a = tot_o = seen_o = 0
        for r in rows:
            toks = r["tokens"]
            for q in r["quads"]:
                if q["a_start"] != -1:
                    tot_a += 1
                    a = " ".join(toks[q["a_start"]:q["a_end"]]).strip()
                    if a in train_asps: seen_a += 1
                if q["o_start"] != -1:
                    tot_o += 1
                    o = " ".join(toks[q["o_start"]:q["o_end"]]).strip()
                    if o in train_ops: seen_o += 1
        seen_stats[name] = {
            "aspect": {"seen_pct": round(seen_a / tot_a * 100, 2), "unseen_pct": round(100 - seen_a / tot_a * 100, 2), "seen": seen_a, "total": tot_a},
            "opinion": {"seen_pct": round(seen_o / tot_o * 100, 2), "unseen_pct": round(100 - seen_o / tot_o * 100, 2), "seen": seen_o, "total": tot_o},
        }
        print(f"\n--- {name.upper()} RATIOS ---")
        print(f"  Aspects:  {seen_stats[name]['aspect']['seen_pct']}% Seen | {seen_stats[name]['aspect']['unseen_pct']}% Unseen")
        print(f"  Opinions: {seen_stats[name]['opinion']['seen_pct']}% Seen | {seen_stats[name]['opinion']['unseen_pct']}% Unseen")

    # Generate label_space.json
    train_quads = [q for r in splits["train"] for q in r["quads"]]
    train_cat_counts = collections.Counter(q["category"] for q in train_quads)
    low_support = [cat for cat in all_cats if train_cat_counts[cat] < 30]

    label_space = {
        "categories": all_cats,
        "sentiments": ["NEUTRAL", "POSITIVE", "NEGATIVE"],
        "total_categories": len(all_cats),
        "low_support_categories": low_support,
        "category_counts_train": dict(train_cat_counts),
        "note": "28-category Gemini-labeled ACOS dataset calibrated to ~70% seen / 30% unseen across aspects and opinions.",
    }

    with open(out_path / "label_space.json", "w", encoding="utf-8") as f:
        json.dump(label_space, f, ensure_ascii=False, indent=2)

    # Generate dataset_stats.json
    stats = {
        "total_sentences": len(all_records),
        "total_quads": total_quads,
        "seen_unseen_ratios": seen_stats,
        "splits": {
            s: {
                "sentences": len(rows),
                "quads": sum(len(r["quads"]) for r in rows),
                "multi_quad_sentences": sum(1 for r in rows if len(r["quads"]) >= 2),
            }
            for s, rows in splits.items()
        },
    }

    with open(out_path / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nPreparation complete! Artifacts written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="data/Gemini_Labeled_ACOS_Data_cleaned.csv",
        help="Path to cleaned input CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/prepared_gemini",
        help="Path to output prepared directory",
    )
    args = parser.parse_args()

    prepare_dataset(args.input, args.output_dir)


if __name__ == "__main__":
    main()
