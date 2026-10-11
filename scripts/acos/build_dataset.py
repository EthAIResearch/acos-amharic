"""Dataset assembler and validator for the clean Amharic ACOS benchmark dataset
covering all 23 mandated categories with strong multi-quadruple prevalence.
"""

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

CATEGORIES_23 = [
    "ECONOMY#COMMUNITY_SUPPORT",
    "ECONOMY#EMPLOYMENT",
    "ECONOMY#TAXATION",
    "ECONOMY#UTILITIES",
    "GOVERNANCE#CITIZEN_ENGAGEMENT",
    "GOVERNANCE#COMMUNITY_SUPPORT",
    "GOVERNANCE#LEGISLATION",
    "GOVERNANCE#TRANSPARENCY",
    "HEALTHCARE#GENERAL",
    "INFRASTRUCTURE#TRANSPORTATION",
    "INFRASTRUCTURE#UTILITIES",
    "PUBLIC_SAFETY#CRIME",
    "PUBLIC_SAFETY#CRIME_SERVICES",
    "PUBLIC_SAFETY#EMERGENCY_SERVICES",
    "PUBLIC_SERVICES#COMMUNITY_SUPPORT",
    "PUBLIC_SERVICES#EDUCATION",
    "PUBLIC_SERVICES#EMERGENCY_SERVICES",
    "PUBLIC_SERVICES#HEALTHCARE",
    "PUBLIC_SERVICES#INFRASTRUCTURE",
    "PUBLIC_SERVICES#UTILITIES",
    "SOCIAL#COMMUNITY_SUPPORT",
    "SOCIAL#EDUCATION",
    "SOCIAL#EQUALITY_JUSTICE",
]

SENTIMENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]


def quad_key(q: dict) -> tuple:
    return (q["a_start"], q["a_end"], q["category"], q["sentiment"], q["o_start"], q["o_end"])


def validate_sentence(rec: dict) -> bool:
    """Verify that token spans and schemas are strictly valid."""
    tokens = rec.get("tokens", [])
    quads = rec.get("quads", [])
    if not tokens or not quads:
        return False

    N = len(tokens)
    for q in quads:
        a_s, a_e = q["a_start"], q["a_end"]
        o_s, o_e = q["o_start"], q["o_end"]

        # Validate category and sentiment
        if q["category"] not in CATEGORIES_23:
            return False
        if q["sentiment"] not in SENTIMENTS:
            return False

        # Validate aspect span
        if a_s != -1 or a_e != -1:
            if not (0 <= a_s < a_e <= N):
                return False

        # Validate opinion span
        if o_s != -1 or o_e != -1:
            if not (0 <= o_s < o_e <= N):
                return False

    return True


def deduplicate_and_sort(records: List[dict]) -> List[dict]:
    """Deduplicate records by normalized text, remove duplicate quads inside each record."""
    by_text = {}
    for r in records:
        text_norm = " ".join(r["text"].strip().split())
        if text_norm not in by_text:
            by_text[text_norm] = {
                "text": text_norm,
                "tokens": text_norm.split(),
                "quads": [],
            }
        seen_keys = {quad_key(q) for q in by_text[text_norm]["quads"]}
        for q in r["quads"]:
            qk = quad_key(q)
            if qk not in seen_keys:
                seen_keys.add(qk)
                by_text[text_norm]["quads"].append(q)

    # Sort quads canonically
    valid_records = []
    for r in by_text.values():
        if validate_sentence(r):
            # Sort quads: explicit aspects first, then explicit opinions, then implicit
            r["quads"].sort(key=lambda q: (
                0 if q["a_start"] != -1 else (1 if q["o_start"] != -1 else 2),
                q["a_start"],
                q["o_start"],
                q["category"],
            ))
            valid_records.append(r)

    return valid_records


def build_and_split_dataset(
    annotated_file: Path,
    output_dir: Path,
    target_multi_quad_ratio: float = 0.65,
    train_ratio: float = 0.80,
    dev_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    raw_records = []
    with open(annotated_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    print(f"[Dataset] Loaded {len(raw_records)} annotated records.")
    cleaned = deduplicate_and_sort(raw_records)
    print(f"[Dataset] Valid unique sentences: {len(cleaned)}")

    # Separate multi-quad and single-quad sentences
    multi_quad = [r for r in cleaned if len(r["quads"]) >= 2]
    single_quad = [r for r in cleaned if len(r["quads"]) == 1]

    print(f"[Dataset] Available Multi-quad sentences: {len(multi_quad)}")
    print(f"[Dataset] Available Single-quad sentences: {len(single_quad)}")

    # Sort single quads to prioritize rarer categories first
    cat_counts_multi = collections.Counter(q["category"] for r in multi_quad for q in r["quads"])
    single_quad.sort(key=lambda r: cat_counts_multi.get(r["quads"][0]["category"], 999))

    # Filter to guarantee majority multi-quad prevalence (>= target_multi_quad_ratio, e.g. 65%)
    max_single = int(len(multi_quad) * ((1.0 - target_multi_quad_ratio) / target_multi_quad_ratio))
    selected_single = single_quad[:max_single] if len(single_quad) > max_single else single_quad

    selected_records = multi_quad + selected_single
    total_sents = len(selected_records)
    print(f"[Dataset] Selected total sentences: {total_sents} (multi: {len(multi_quad)}, single: {len(selected_single)})")

    # Category-aware stratified split preserving coverage across all 23 categories
    random.seed(seed)
    cat_buckets = collections.defaultdict(list)
    for r in selected_records:
        rarest = min(r["quads"], key=lambda q: cat_counts_multi.get(q["category"], 999))["category"]
        cat_buckets[rarest].append(r)

    train_set, dev_set, test_set = [], [], []

    for cat in sorted(cat_buckets):
        rows = cat_buckets[cat]
        random.shuffle(rows)
        N = len(rows)
        if N == 1:
            train_set.append(rows[0])
        elif N == 2:
            train_set.append(rows[0])
            test_set.append(rows[1])
        elif N == 3:
            train_set.append(rows[0])
            train_set.append(rows[1])
            test_set.append(rows[2])
        else:
            n_dev = max(1, int(round(N * dev_ratio)))
            n_test = max(1, int(round(N * test_ratio)))
            n_train = N - n_dev - n_test
            train_set.extend(rows[:n_train])
            dev_set.extend(rows[n_train : n_train + n_dev])
            test_set.extend(rows[n_train + n_dev:])

    random.shuffle(train_set)
    random.shuffle(dev_set)
    random.shuffle(test_set)

    splits = {
        "train": train_set,
        "dev": dev_set,
        "test": test_set,
    }

    # Write splits
    for name, rows in splits.items():
        split_path = output_dir / f"{name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Dataset] Wrote {name}: {len(rows)} sentences ({sum(len(r['quads']) for r in rows)} quads)")

    # Build and write label_space.json
    all_quads = [q for r in selected_records for q in r["quads"]]
    cat_counts = collections.Counter(q["category"] for q in all_quads)
    sent_counts = collections.Counter(q["sentiment"] for q in all_quads)

    quad_dist = collections.Counter(len(r["quads"]) for r in selected_records)

    label_space = {
        "categories": sorted(CATEGORIES_23),
        "sentiments": SENTIMENTS,
        "total_categories": len(CATEGORIES_23),
        "category_counts": dict(cat_counts),
        "note": "Complete 23-category externally-mandated taxonomy with guaranteed multi-quadruple prevalence from authentic Amharic book corpus.",
    }
    with open(output_dir / "label_space.json", "w", encoding="utf-8") as f:
        json.dump(label_space, f, ensure_ascii=False, indent=2)

    stats = {
        "total_sentences": total_sents,
        "total_quadruples": len(all_quads),
        "average_quads_per_sentence": round(len(all_quads) / total_sents, 2) if total_sents else 0,
        "quadruple_count_distribution": {
            f"{k}_quads": quad_dist[k] for k in sorted(quad_dist)
        },
        "multi_quad_percentage": round(100.0 * sum(quad_dist[k] for k in quad_dist if k >= 2) / total_sents, 1) if total_sents else 0,
        "sentiment_distribution": dict(sent_counts),
        "category_distribution": dict(cat_counts),
        "aspect_types": {
            "explicit": sum(1 for q in all_quads if q["a_start"] != -1),
            "implicit": sum(1 for q in all_quads if q["a_start"] == -1),
        },
        "opinion_types": {
            "explicit": sum(1 for q in all_quads if q["o_start"] != -1),
            "implicit": sum(1 for q in all_quads if q["o_start"] == -1),
        },
        "splits": {
            name: {
                "sentences": len(rows),
                "quadruples": sum(len(r["quads"]) for r in rows),
                "multi_quad_sentences": sum(1 for r in rows if len(r["quads"]) >= 2),
            }
            for name, rows in splits.items()
        },
    }

    with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[Dataset] Generation complete! Statistics saved to {output_dir / 'dataset_stats.json'}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble clean ACOS dataset")
    parser.add_argument("--annotated", type=str, default="data/book_acos/annotated_grounded.jsonl")
    parser.add_argument("--out-dir", type=str, default="data/book_acos")
    parser.add_argument("--multi-ratio", type=float, default=0.65)
    args = parser.parse_args()

    build_and_split_dataset(
        annotated_file=Path(args.annotated),
        output_dir=Path(args.out_dir),
        target_multi_quad_ratio=args.multi_ratio,
    )
