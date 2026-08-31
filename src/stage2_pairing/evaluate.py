"""
Stage 2 evaluation: heuristic aspect-opinion pairing against gold spans.

Usage:
    python evaluate.py --data ../../data/prepared/test.jsonl

Note: this evaluates pairing IN ISOLATION using gold (not predicted) spans --
it measures whether the pairing logic itself is correct, not end-to-end
pipeline performance. Real pipeline F1 will additionally be bottlenecked by
Stage 1's span extraction errors (a missed or wrong span means no correct
pair is possible regardless of pairing quality) -- that combined number
belongs in Stage 6's end-to-end evaluation, not here.
"""
import argparse
import json

from candidates import explicit_quads, heuristic_pairs


def evaluate(path: str) -> dict:
    tp = fp = fn = 0
    n_sentences = 0

    with open(path, encoding="utf-8") as data:
        for line in data:
            rec = json.loads(line)
            ex_quads = explicit_quads(rec["quads"])
            if not ex_quads:
                continue
            n_sentences += 1

            aspect_spans = sorted({(q["a_start"], q["a_end"]) for q in ex_quads})
            opinion_spans = sorted({(q["o_start"], q["o_end"]) for q in ex_quads})
            true_pairs = {(q["a_start"], q["a_end"], q["o_start"], q["o_end"]) for q in ex_quads}

            pred_pairs = heuristic_pairs(aspect_spans, opinion_spans)

            tp += len(pred_pairs & true_pairs)
            fp += len(pred_pairs - true_pairs)
            fn += len(true_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "sentences_evaluated": n_sentences,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    metrics = evaluate(args.data)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
