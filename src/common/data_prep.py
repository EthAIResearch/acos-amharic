"""
Data preparation for Amharic ACOS (Aspect-Category-Opinion-Sentiment) extraction.

Input format (TSV), one sentence per line:
    <text>\t<quad1>\t<quad2>...
where each quad is a single field, space-separated:
    "aStart,aEnd CATEGORY sentiment oStart,oEnd"
Spans are whitespace-token indices into `text.split()`. "-1,-1" = implicit (no span).
Sentiment: 0=neutral, 1=positive, 2=negative.

This script:
  1. Parses raw TSV into structured records.
  2. Canonicalizes the category schema (fixes cross-domain label overlap).
  3. Writes clean JSONL files (train/dev/test) ready for tokenizer alignment.
  4. Prints before/after category distribution so the mapping is auditable.
"""
import json
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

SENTIMENT_MAP = {"0": "NEUTRAL", "1": "POSITIVE", "2": "NEGATIVE"}

# Manual merges for near-duplicate fine-grained tags that don't cleanly collapse
# by dropping the domain prefix alone.
MANUAL_CATEGORY_MERGE = {
    "CRIME_SERVICES": "CRIME",
    "INFRASTRUCTURE": "UTILITIES",  # from PUBLIC_SERVICES#INFRASTRUCTURE (1 example)
}

MIN_CATEGORY_COUNT = 30  # categories with fewer than this many train instances -> OTHER


@dataclass
class Quad:
    a_start: int
    a_end: int
    category: str
    sentiment: str
    o_start: int
    o_end: int

    @property
    def aspect_implicit(self) -> bool:
        return self.a_start == -1

    @property
    def opinion_implicit(self) -> bool:
        return self.o_start == -1


@dataclass
class Example:
    text: str
    tokens: list[str]
    quads: list[Quad]


def parse_quad_field(field: str) -> Quad:
    a_span, cat, sent, o_span = field.strip().split(" ")
    a_start, a_end = (int(x) for x in a_span.split(","))
    o_start, o_end = (int(x) for x in o_span.split(","))
    return Quad(a_start, a_end, cat, SENTIMENT_MAP[sent], o_start, o_end)


def parse_tsv(path: str) -> list[Example]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            text = parts[0]
            tokens = text.split()
            quads = [parse_quad_field(q) for q in parts[1:] if q.strip()]
            examples.append(Example(text=text, tokens=tokens, quads=quads))
    return examples


def canonical_category(raw_cat: str) -> str:
    """Drop the domain prefix (e.g. GOVERNANCE#TRANSPARENCY -> TRANSPARENCY),
    which merges tags that were duplicated across domains due to schema overlap."""
    fine = raw_cat.split("#", 1)[1] if "#" in raw_cat else raw_cat
    return MANUAL_CATEGORY_MERGE.get(fine, fine)


def build_category_mapping(train_examples: list[Example]) -> dict:
    """Build raw_category -> final_category mapping, applying the MIN_CATEGORY_COUNT
    threshold (computed on TRAIN only, then reused for dev/test for consistency)."""
    counts = Counter()
    raw_to_canonical = {}
    for ex in train_examples:
        for q in ex.quads:
            canon = canonical_category(q.category)
            raw_to_canonical[q.category] = canon
            counts[canon] += 1

    final_mapping = {}
    for raw_cat, canon in raw_to_canonical.items():
        final_mapping[raw_cat] = canon if counts[canon] >= MIN_CATEGORY_COUNT else "OTHER"
    return final_mapping


def apply_mapping(examples: list[Example], mapping: dict) -> list[Example]:
    for ex in examples:
        for q in ex.quads:
            q.category = mapping.get(q.category, "OTHER")
    return examples


def report_distribution(examples: list[Example], label: str):
    counts = Counter(q.category for ex in examples for q in ex.quads)
    total = sum(counts.values())
    print(f"\n--- {label}: category distribution after canonicalization ({total} quads) ---")
    for cat, c in counts.most_common():
        print(f"  {cat}: {c} ({100*c/total:.1f}%)")


def write_jsonl(examples: list[Example], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            rec = {
                "text": ex.text,
                "tokens": ex.tokens,
                "quads": [asdict(q) for q in ex.quads],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--out_dir", default="./prepared")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    train_examples = parse_tsv(args.train)
    test_examples = parse_tsv(args.test)
    dev_examples = parse_tsv(args.dev) if args.dev else None

    mapping = build_category_mapping(train_examples)

    print("=== Category mapping (raw -> final) ===")
    for raw, final in sorted(mapping.items()):
        flag = "  <-- merged/renamed" if raw.split("#", 1)[-1] != final else ""
        print(f"  {raw}  ->  {final}{flag}")

    train_examples = apply_mapping(train_examples, mapping)
    test_examples = apply_mapping(test_examples, mapping)
    if dev_examples:
        dev_examples = apply_mapping(dev_examples, mapping)

    report_distribution(train_examples, "TRAIN")
    report_distribution(test_examples, "TEST")

    write_jsonl(train_examples, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(test_examples, os.path.join(args.out_dir, "test.jsonl"))
    if dev_examples:
        write_jsonl(dev_examples, os.path.join(args.out_dir, "dev.jsonl"))

    final_categories = sorted(set(mapping.values()))
    with open(os.path.join(args.out_dir, "label_space.json"), "w", encoding="utf-8") as f:
        json.dump({
            "categories": final_categories,
            "sentiments": ["NEUTRAL", "POSITIVE", "NEGATIVE"],
            "category_mapping": mapping,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nWrote prepared data to {args.out_dir}/  ({len(final_categories)} final categories)")


if __name__ == "__main__":
    main()
