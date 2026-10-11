"""Assemble the enriched civic commentary dataset by merging high-density multi-quad
re-annotations into the base civic dataset (data/prepared/), validating all token spans,
and exporting balanced splits with full 23-category coverage.
"""

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acos.annotate_and_ground import CATEGORIES_23, SENTIMENTS


def norm_text(t: str) -> str:
    return " ".join(t.strip().split())


def quad_key(q: dict) -> tuple:
    return (q["a_start"], q["a_end"], q["category"], q["sentiment"], q["o_start"], q["o_end"])


def validate_record(rec: dict) -> bool:
    tokens = rec.get("tokens", [])
    quads = rec.get("quads", [])
    if not tokens or not quads:
        return False
    N = len(tokens)
    for q in quads:
        if q["category"] not in CATEGORIES_23 or q["sentiment"] not in SENTIMENTS:
            return False
        a_s, a_e = q["a_start"], q["a_end"]
        o_s, o_e = q["o_start"], q["o_end"]
        if a_s != -1 and not (0 <= a_s < a_e <= N):
            return False
        if o_s != -1 and not (0 <= o_s < o_e <= N):
            return False
    return True


def assemble_dataset(
    prepared_dir: Path,
    reannotated_file: Path,
    output_dir: Path,
) -> Dict[str, any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all re-annotated records
    reannotated_map: Dict[str, dict] = {}
    if reannotated_file.exists():
        with open(reannotated_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if validate_record(r):
                        reannotated_map[norm_text(r["text"])] = r

    print(f"[Assembler] Loaded {len(reannotated_map)} valid re-annotated civic sentences.")

    splits_data = {}
    stats_by_split = {}
    overall_cat_counter = collections.Counter()
    overall_sent_counter = collections.Counter()
    overall_quad_dist = collections.Counter()
    total_sentences = 0
    total_quads = 0

    for split in ["train", "dev", "test"]:
        src_path = prepared_dir / f"{split}.jsonl"
        out_rows = []
        replaced_count = 0

        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                k = norm_text(rec["text"])

                # If we have an enriched re-annotation, use it!
                if k in reannotated_map:
                    enriched = reannotated_map[k]
                    rec["quads"] = enriched["quads"]
                    replaced_count += 1

                # Clean duplicate quads inside the record
                seen_q = set()
                deduped_q = []
                for q in rec["quads"]:
                    qk = quad_key(q)
                    if qk not in seen_q and q["category"] in CATEGORIES_23:
                        seen_q.add(qk)
                        deduped_q.append(q)

                rec["quads"] = deduped_q
                if validate_record(rec):
                    out_rows.append(rec)

        splits_data[split] = out_rows
        dst_path = output_dir / f"{split}.jsonl"
        with open(dst_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Stats for this split
        split_quads = [q for r in out_rows for q in r["quads"]]
        split_cats = collections.Counter(q["category"] for q in split_quads)
        split_sents = collections.Counter(q["sentiment"] for q in split_quads)
        multi_cnt = sum(1 for r in out_rows if len(r["quads"]) >= 2)

        stats_by_split[split] = {
            "sentences": len(out_rows),
            "quadruples": len(split_quads),
            "multi_quad_sentences": multi_cnt,
            "multi_quad_percentage": round(100.0 * multi_cnt / len(out_rows), 2) if out_rows else 0,
            "enriched_replacements": replaced_count,
            "categories_present": len(split_cats),
        }

        overall_cat_counter.update(split_cats)
        overall_sent_counter.update(split_sents)
        for r in out_rows:
            overall_quad_dist[len(r["quads"])] += 1
        total_sentences += len(out_rows)
        total_quads += len(split_quads)

        print(f"[Assembler] {split}: {len(out_rows)} sents ({multi_cnt} multi-quad, {replaced_count} enriched), {len(split_quads)} quads, {len(split_cats)}/23 categories")

    # Write label_space.json
    label_space = {
        "categories": sorted(CATEGORIES_23),
        "sentiments": sorted(list(SENTIMENTS)),
        "total_categories": len(CATEGORIES_23),
        "category_counts": dict(overall_cat_counter),
        "note": "Complete 23-category taxonomy covering all domains on enriched authentic civic commentary.",
    }
    with open(output_dir / "label_space.json", "w", encoding="utf-8") as f:
        json.dump(label_space, f, ensure_ascii=False, indent=2)

    # Write dataset_stats.json
    multi_total = sum(overall_quad_dist[k] for k in overall_quad_dist if k >= 2)
    stats = {
        "total_sentences": total_sentences,
        "total_quadruples": total_quads,
        "average_quads_per_sentence": round(total_quads / total_sentences, 2) if total_sentences else 0,
        "quadruple_count_distribution": {
            f"{k}_quads": overall_quad_dist[k] for k in sorted(overall_quad_dist)
        },
        "multi_quad_percentage": round(100.0 * multi_total / total_sentences, 2) if total_sentences else 0,
        "sentiment_distribution": dict(overall_sent_counter),
        "category_distribution": dict(overall_cat_counter),
        "splits": stats_by_split,
    }
    with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[Assembler] Enriched dataset generation complete in {output_dir}!")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble enriched civic dataset")
    parser.add_argument("--prepared-dir", type=str, default="data/prepared")
    parser.add_argument("--reannotated", type=str, default="data/prepared_reannotated.jsonl")
    parser.add_argument("--out-dir", type=str, default="data/prepared_enriched")
    args = parser.parse_args()

    assemble_dataset(
        prepared_dir=Path(args.prepared_dir),
        reannotated_file=Path(args.reannotated),
        output_dir=Path(args.out_dir),
    )
