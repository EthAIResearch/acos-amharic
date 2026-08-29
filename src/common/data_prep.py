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
  2. Writes clean JSONL files (train/dev/test) ready for tokenizer alignment.
  3. Reports category counts as-is -- for visibility only. Category labels are
     the fixed, externally-mandated taxonomy and are NEVER renamed, merged, or
     dropped by this script. Low-count categories are flagged in the printed
     report and in label_space.json ("low_support") purely as a heads-up for
     modeling strategy (e.g. class weighting) -- the label itself is untouched.
"""
import json
import argparse
import os
from collections import Counter
from dataclasses import dataclass, asdict

SENTIMENT_MAP = {"0": "NEUTRAL", "1": "POSITIVE", "2": "NEGATIVE"}

# Below this many train instances, a category is flagged as low-support in the
# report and in label_space.json. This does NOT change the label in any way --
# it's a signal for choosing a training strategy (class weighting, focal loss,
# oversampling), never a reason to rename or merge the category.
LOW_SUPPORT_THRESHOLD = 30


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


def report_distribution(examples: list[Example], label: str) -> Counter:
    counts = Counter(q.category for ex in examples for q in ex.quads)
    total = sum(counts.values())
    print(f"\n--- {label}: category distribution ({total} quads, {len(counts)} categories) ---")
    for cat, c in counts.most_common():
        flag = "  <-- LOW SUPPORT" if c < LOW_SUPPORT_THRESHOLD else ""
        print(f"  {cat}: {c} ({100*c/total:.1f}%){flag}")
    return counts


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

    os.makedirs(args.out_dir, exist_ok=True)

    train_examples = parse_tsv(args.train)
    test_examples = parse_tsv(args.test)
    dev_examples = parse_tsv(args.dev) if args.dev else None

    train_counts = report_distribution(train_examples, "TRAIN")
    test_counts = report_distribution(test_examples, "TEST")
    if dev_examples:
        report_distribution(dev_examples, "DEV")

    write_jsonl(train_examples, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(test_examples, os.path.join(args.out_dir, "test.jsonl"))
    if dev_examples:
        write_jsonl(dev_examples, os.path.join(args.out_dir, "dev.jsonl"))

    # The fixed category taxonomy, exactly as given -- union of every raw label
    # seen in train (the split that should define the label space).
    all_categories = sorted(train_counts)
    low_support = [c for c in all_categories if train_counts[c] < LOW_SUPPORT_THRESHOLD]
    zero_train_but_in_test = sorted(set(test_counts) - set(train_counts))

    with open(os.path.join(args.out_dir, "label_space.json"), "w", encoding="utf-8") as f:
        json.dump({
            "categories": all_categories,
            "sentiments": ["NEUTRAL", "POSITIVE", "NEGATIVE"],
            "low_support_categories": low_support,
            "note": (
                "Categories are the fixed, externally-mandated taxonomy and are "
                "used exactly as given -- never renamed or merged. "
                "'low_support_categories' lists categories with fewer than "
                f"{LOW_SUPPORT_THRESHOLD} train examples; use class weighting / "
                "focal loss / oversampling for these during training, not "
                "relabeling."
            ),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nWrote prepared data to {args.out_dir}/  ({len(all_categories)} categories, unchanged)")
    if low_support:
        print(f"Low-support categories (<{LOW_SUPPORT_THRESHOLD} train examples): {low_support}")
    if zero_train_but_in_test:
        print(f"WARNING -- appear in TEST with ZERO train examples (unlearnable as-is): {zero_train_but_in_test}")


if __name__ == "__main__":
    main()
