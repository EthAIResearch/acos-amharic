"""Mine candidate sentences from the existing civic commentary dataset (data/prepared/)
for multi-quadruple re-annotation and rare-category representation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acos.mine_sentences import CATEGORY_LEXICONS, CONNECTORS, match_categories

RARE_CATEGORIES = [
    "ECONOMY#UTILITIES",
    "PUBLIC_SERVICES#UTILITIES",
    "GOVERNANCE#COMMUNITY_SUPPORT",
    "PUBLIC_SERVICES#COMMUNITY_SUPPORT",
    "PUBLIC_SERVICES#INFRASTRUCTURE",
    "PUBLIC_SERVICES#HEALTHCARE",
    "PUBLIC_SERVICES#EMERGENCY_SERVICES",
    "SOCIAL#EDUCATION",
    "INFRASTRUCTURE#UTILITIES",
    "PUBLIC_SAFETY#EMERGENCY_SERVICES",
]


def is_candidate(rec: dict) -> bool:
    text = rec["text"]
    tokens = rec["tokens"]
    quads = rec.get("quads", [])

    # Keep sentences between 6 and 80 tokens (civic commentary range)
    if not (6 <= len(tokens) <= 80):
        return False

    matched_cats = match_categories(text)

    # 1. Any sentence matching rare categories (must re-annotate to guarantee support)
    if any(rc in matched_cats for rc in RARE_CATEGORIES):
        return True

    # 2. Sentences with contrastive connectors that currently have only 1 quad
    has_conn = any(c in text for c in CONNECTORS)
    if has_conn and len(quads) <= 1:
        return True

    # 3. Sentences matching 2 or more category lexicons
    if len(matched_cats) >= 2 and len(quads) <= 1:
        return True

    return False


def mine_candidates(
    prepared_dir: Path,
    output_path: Path,
    max_candidates: int = 3000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_texts: Set[str] = set()
    rare_candidates: List[dict] = []
    compound_candidates: List[dict] = []

    splits = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    for split_file in splits:
        split_path = prepared_dir / split_file
        if not split_path.exists():
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                norm = " ".join(rec["text"].strip().split())
                if norm in seen_texts:
                    continue

                if not (6 <= len(rec["tokens"]) <= 80):
                    continue

                matched = match_categories(rec["text"])
                cand_item = {
                    "text": rec["text"],
                    "tokens": rec["tokens"],
                    "original_quads": rec.get("quads", []),
                    "matched_categories": matched,
                    "source_split": split_file.replace(".jsonl", ""),
                }

                # Priority 1: Any match for rare categories
                if any(rc in matched for rc in RARE_CATEGORIES):
                    rare_candidates.append(cand_item)
                    seen_texts.add(norm)
                elif any(c in rec["text"] for c in CONNECTORS) or len(matched) >= 2:
                    compound_candidates.append(cand_item)
                    seen_texts.add(norm)

    print(f"[Miner] Found {len(rare_candidates)} rare-category candidates and {len(compound_candidates)} compound candidates")
    
    # Combine: ALL rare candidates + compound candidates up to max_candidates
    candidates = rare_candidates + compound_candidates[: max(0, max_candidates - len(rare_candidates))]
    print(f"[Miner] Total candidates selected: {len(candidates)}")

    cat_counts = {}
    for c in candidates:
        for cat in c["matched_categories"]:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("\n[Miner] Final Candidate Category Matches:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {cnt}")

    with open(output_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n[Miner] Saved candidates to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine civic candidates for re-annotation")
    parser.add_argument("--prepared-dir", type=str, default="data/prepared")
    parser.add_argument("--out", type=str, default="data/prepared_candidates.jsonl")
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()

    mine_candidates(
        prepared_dir=Path(args.prepared_dir),
        output_path=Path(args.out),
        max_candidates=args.limit,
    )
