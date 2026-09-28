"""Clean data/prepared/ into data/prepared_clean/ WITHOUT touching the category schema.

Fixes applied (categories are never merged, renamed, or dropped):
1. Merge duplicate texts within each split (union of their quads, exact-duplicate quads removed).
2. Resolve same-(aspect,opinion)-span sentiment contradictions inside a sentence
   using the train-set majority sentiment for that surface pair (fallback: first seen).
3. Remove train texts that leak into dev/test, and dev texts that leak into test.
4. Drop under-annotated over-long train texts (> --max-train-tokens, default 100).

Usage: python3 data/clean_dataset.py [--max-train-tokens N] [--keep-long]
"""

import argparse
import collections
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "prepared"
DST = ROOT / "prepared_clean"


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())


def quad_key(q: dict) -> tuple:
    return (q["a_start"], q["a_end"], q["category"], q["sentiment"], q["o_start"], q["o_end"])


def span_text(tokens: list, s: int, e: int):
    return " ".join(tokens[s:e]) if s != -1 else None


def load(split: str) -> list:
    with open(SRC / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_pair_sentiment_majority(rows: list) -> dict:
    votes = collections.defaultdict(collections.Counter)
    for r in rows:
        for q in r["quads"]:
            pair = (span_text(r["tokens"], q["a_start"], q["a_end"]),
                    span_text(r["tokens"], q["o_start"], q["o_end"]))
            votes[pair][q["sentiment"]] += 1
    majority = {}
    for pair, c in votes.items():
        top = c.most_common()
        # only a clear majority counts; ties fall back to first-seen at use site
        if len(top) == 1 or top[0][1] > top[1][1]:
            majority[pair] = top[0][0]
    return majority


def merge_duplicates(rows: list, stats: collections.Counter) -> list:
    by_text = {}
    order = []
    for r in rows:
        key = norm_text(r["text"])
        if key not in by_text:
            by_text[key] = {"text": r["text"], "tokens": r["tokens"], "quads": []}
            order.append(key)
        else:
            stats["duplicate_rows_merged"] += 1
        seen = {quad_key(q) for q in by_text[key]["quads"]}
        for q in r["quads"]:
            if quad_key(q) in seen:
                stats["duplicate_quads_dropped"] += 1
            else:
                seen.add(quad_key(q))
                by_text[key]["quads"].append(dict(q))
    return [by_text[k] for k in order]


def resolve_sentiment_conflicts(rows: list, majority: dict, stats: collections.Counter) -> None:
    for r in rows:
        groups = collections.defaultdict(list)
        for q in r["quads"]:
            groups[(q["a_start"], q["a_end"], q["o_start"], q["o_end"])].append(q)
        for (a_s, a_e, o_s, o_e), quads in groups.items():
            sentiments = {q["sentiment"] for q in quads}
            if len(sentiments) <= 1:
                continue
            pair = (span_text(r["tokens"], a_s, a_e), span_text(r["tokens"], o_s, o_e))
            chosen = majority.get(pair, quads[0]["sentiment"])
            if chosen not in sentiments:
                chosen = quads[0]["sentiment"]
            for q in quads:
                if q["sentiment"] != chosen:
                    q["sentiment"] = chosen
                    stats["sentiment_conflicts_resolved"] += 1
        # re-dedup: conflict resolution can make quads identical
        seen, kept = set(), []
        for q in r["quads"]:
            if quad_key(q) in seen:
                stats["duplicate_quads_dropped"] += 1
            else:
                seen.add(quad_key(q))
                kept.append(q)
        r["quads"] = kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-tokens", type=int, default=100)
    parser.add_argument("--keep-long", action="store_true",
                        help="keep over-long train texts instead of dropping them")
    args = parser.parse_args()

    splits = {s: load(s) for s in ("train", "dev", "test")}
    majority = build_pair_sentiment_majority(splits["train"])
    stats = {s: collections.Counter() for s in splits}

    for s in splits:
        splits[s] = merge_duplicates(splits[s], stats[s])
        resolve_sentiment_conflicts(splits[s], majority, stats[s])

    test_texts = {norm_text(r["text"]) for r in splits["test"]}
    dev_texts = {norm_text(r["text"]) for r in splits["dev"]}

    kept = []
    for r in splits["train"]:
        key = norm_text(r["text"])
        if key in test_texts or key in dev_texts:
            stats["train"]["leaked_rows_removed"] += 1
        elif not args.keep_long and len(r["tokens"]) > args.max_train_tokens:
            stats["train"]["long_rows_dropped"] += 1
        else:
            kept.append(r)
    splits["train"] = kept

    kept_dev = []
    for r in splits["dev"]:
        if norm_text(r["text"]) in test_texts:
            stats["dev"]["leaked_rows_removed"] += 1
        else:
            kept_dev.append(r)
    splits["dev"] = kept_dev

    DST.mkdir(exist_ok=True)
    shutil.copy(SRC / "label_space.json", DST / "label_space.json")
    report = {"source": str(SRC), "max_train_tokens": None if args.keep_long else args.max_train_tokens}
    for s, rows in splits.items():
        with open(DST / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[s] = {"sentences": len(rows),
                     "quads": sum(len(r["quads"]) for r in rows),
                     **dict(stats[s])}
        print(f"{s}: {report[s]}")
    with open(DST / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten to {DST}")


if __name__ == "__main__":
    main()
