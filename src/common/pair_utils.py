"""
Shared utilities for span-pair classification tasks (Stage 3: category,
Stage 4: sentiment). Both stages classify a (sentence, aspect_span,
opinion_span) triple -- only the label field and output classes differ.
"""


def explicit_quads(quads: list[dict]) -> list[dict]:
    """Quads where both aspect and opinion are explicit spans (not -1,-1).
    Stage 3/4 only train on these -- implicit cases are Stage 5's job."""
    return [q for q in quads if q["a_start"] != -1 and q["o_start"] != -1]


def word_span_to_subword_range(word_ids: list, start: int, end: int) -> tuple[int, int] | None:
    """Map a word-level span [start, end) to the (min, max) subword token
    indices covering it, using a fast tokenizer's word_ids(). Returns None
    if the span falls entirely outside the tokenizer's truncation window
    (rare -- only affects sentences longer than max_length)."""
    positions = [i for i, wid in enumerate(word_ids) if wid is not None and start <= wid < end]
    if not positions:
        return None
    return min(positions), max(positions)


def build_span_mask(seq_len: int, subword_range: tuple[int, int] | None) -> list[int]:
    """Binary mask over subword positions, 1 for tokens inside the span."""
    mask = [0] * seq_len
    if subword_range is None:
        return mask
    lo, hi = subword_range
    for i in range(lo, hi + 1):
        mask[i] = 1
    return mask


def compute_class_weights(label_counts: dict, num_classes: int, cap: float = 15.0) -> dict:
    """Inverse-frequency class weights, capped to avoid runaway weights on
    near-zero-count classes destabilizing training."""
    total = sum(label_counts.values())
    weights = {}
    for label, count in label_counts.items():
        w = total / (num_classes * count) if count > 0 else cap
        weights[label] = min(w, cap)
    return weights


def compute_log_priors(label_counts: dict, num_classes: int) -> list[float]:
    """Empirical log-priors log(P(y)) for logit adjustment (Menon et al., 2021).
    Adds a small epsilon floor so zero-count classes produce valid finite log-priors."""
    import math

    total = sum(label_counts.values())
    priors = []
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        prob = max(count / total, 1e-12) if total > 0 else 1.0 / num_classes
        priors.append(math.log(prob))
    return priors
