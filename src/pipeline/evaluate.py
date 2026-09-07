"""
Stage 6b: exact-match quadruple evaluation.

Standard ACOS metric (Cai et al., 2021): a predicted quad counts as correct
only if aspect span, category, sentiment, and opinion span ALL match gold
exactly (implicit sides match if both are -1,-1). Reports three views, not
just one combined number -- see docs/stage6_evaluation_analysis.md for why
pooling them into a single F1 would hide exactly the comparison this
project's data-driven design decisions depend on:
  - "all": every gold quad, the number that goes in a single headline metric
  - "explicit": only quads where both sides are explicit in gold
  - "implicit": only quads where at least one side is implicit in gold
"""


def is_implicit_quad(quad: dict) -> bool:
    return quad["a_start"] == -1 or quad["o_start"] == -1


def quad_key(quad: dict) -> tuple:
    return (quad["a_start"], quad["a_end"], quad["category"], quad["sentiment"],
            quad["o_start"], quad["o_end"])


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_quads(pred_by_sentence: dict, gold_by_sentence: dict) -> dict:
    """
    pred_by_sentence / gold_by_sentence: {sentence_id: [quad_dict, ...]}
    Sentence IDs must match between the two (use the same iteration order
    / keys the caller used to build both, e.g. line number in the JSONL).

    Returns {"all": {...}, "explicit": {...}, "implicit": {...}}, each a
    precision/recall/F1 dict from _prf().
    """
    counts = {"all": [0, 0, 0], "explicit": [0, 0, 0], "implicit": [0, 0, 0]}  # tp, fp, fn

    for sent_id, gold_quads in gold_by_sentence.items():
        pred_quads = pred_by_sentence.get(sent_id, [])

        for subset, filt in [("all", lambda q: True),
                              ("explicit", lambda q: not is_implicit_quad(q)),
                              ("implicit", is_implicit_quad)]:
            gold_keys = set(quad_key(q) for q in gold_quads if filt(q))
            pred_keys = set(quad_key(q) for q in pred_quads if filt(q))
            tp = len(gold_keys & pred_keys)
            fp = len(pred_keys - gold_keys)
            fn = len(gold_keys - pred_keys)
            counts[subset][0] += tp
            counts[subset][1] += fp
            counts[subset][2] += fn

    return {subset: _prf(*vals) for subset, vals in counts.items()}
