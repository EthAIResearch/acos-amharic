"""
Span-ASTE Pure-Python Utilities
===============================
Implements deterministic span enumeration, bucketing, and label mapping
for Span-ASTE (Xu et al., ACL 2021; arXiv:2107.12214).

Zero PyTorch dependency for clean, deterministic unit-testability.
"""

# Span width and distance bucketing boundaries matching the paper / AllenNLP:
# [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+] -> 10 buckets
BUCKET_RANGES = [
    (0, 0),      # bucket 0
    (1, 1),      # bucket 1
    (2, 2),      # bucket 2
    (3, 3),      # bucket 3
    (4, 4),      # bucket 4
    (5, 7),      # bucket 5
    (8, 15),     # bucket 6
    (16, 31),    # bucket 7
    (32, 63),    # bucket 8
    (64, 10000), # bucket 9
]

# Mention labels for the mention module (Eq. 3)
# 0: INVALID, 1: TARGET (Aspect), 2: OPINION
MENTION2ID = {"INVALID": 0, "TARGET": 1, "OPINION": 2}
ID2MENTION = {0: "INVALID", 1: "TARGET", 2: "OPINION"}

# Relation / sentiment labels for the triplet module (Eq. 6)
# 0: INVALID, 1: POSITIVE, 2: NEGATIVE, 3: NEUTRAL
RELATION2ID = {"INVALID": 0, "POSITIVE": 1, "NEGATIVE": 2, "NEUTRAL": 3}
ID2RELATION = {0: "INVALID", 1: "POSITIVE", 2: "NEGATIVE", 3: "NEUTRAL"}


def bucket_value(val: int) -> int:
    """Map an integer value (e.g. width or distance) into one of 10 bucket indices."""
    if val < 0:
        val = 0
    for idx, (low, high) in enumerate(BUCKET_RANGES):
        if low <= val <= high:
            return idx
    return len(BUCKET_RANGES) - 1


def enumerate_spans(seq_len: int, max_span_length: int = 8) -> list[tuple[int, int]]:
    """
    Enumerate all consecutive spans [s, e) in half-open python slice indexing
    where 0 <= s < e <= seq_len and (e - s) <= max_span_length.
    Returns list of (start, end) tuples.
    """
    spans = []
    for s in range(seq_len):
        for e in range(s + 1, min(s + max_span_length + 1, seq_len + 1)):
            spans.append((s, e))
    return spans


def compute_span_distance(span_a: tuple[int, int], span_b: tuple[int, int]) -> int:
    """
    Computes minimum token distance between two spans (Lee et al., 2017; Xu et al., 2021):
    dist = min(|b_start - a_end|, |a_start - b_end|) if non-overlapping; 0 if overlapping.
    """
    a_s, a_e = span_a
    b_s, b_e = span_b
    if a_s >= b_e:
        return a_s - b_e
    elif b_s >= a_e:
        return b_s - a_e
    else:
        # Overlapping spans have distance 0
        return 0


def build_gold_span_labels(
    spans: list[tuple[int, int]],
    quads: list[dict],
) -> list[int]:
    """
    Maps enumerated spans [s, e) to gold mention classes:
      0: INVALID
      1: TARGET (Aspect)
      2: OPINION
    If a span is both aspect and opinion in rare corner cases, aspect takes priority.
    """
    aspect_spans = set()
    opinion_spans = set()
    for q in quads:
        as_ = q.get("a_start", -1)
        ae = q.get("a_end", -1)
        os_ = q.get("o_start", -1)
        oe = q.get("o_end", -1)
        if as_ != -1 and ae != -1 and as_ < ae:
            aspect_spans.add((as_, ae))
        if os_ != -1 and oe != -1 and os_ < oe:
            opinion_spans.add((os_, oe))

    labels = []
    for span in spans:
        if span in aspect_spans:
            labels.append(MENTION2ID["TARGET"])
        elif span in opinion_spans:
            labels.append(MENTION2ID["OPINION"])
        else:
            labels.append(MENTION2ID["INVALID"])
    return labels


def build_gold_pair_labels(
    target_spans: list[tuple[int, int]],
    opinion_spans: list[tuple[int, int]],
    quads: list[dict],
) -> list[list[int]]:
    """
    For a given candidate target pool and opinion pool, constructs a 2D matrix
    of shape (num_targets, num_opinions) with relation IDs:
      0: INVALID
      1: POSITIVE
      2: NEGATIVE
      3: NEUTRAL
    """
    gold_relations = {}
    for q in quads:
        as_ = q.get("a_start", -1)
        ae = q.get("a_end", -1)
        os_ = q.get("o_start", -1)
        oe = q.get("o_end", -1)
        senti = q.get("sentiment", "").upper()
        if as_ != -1 and os_ != -1:
            rel_id = RELATION2ID.get(senti, RELATION2ID["INVALID"])
            gold_relations[((as_, ae), (os_, oe))] = rel_id

    matrix = []
    for t_span in target_spans:
        row = []
        for o_span in opinion_spans:
            rel = gold_relations.get((t_span, o_span), RELATION2ID["INVALID"])
            row.append(rel)
        matrix.append(row)
    return matrix


def extract_explicit_triplets(quads: list[dict]) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    """
    Extracts all explicit triplets: ((a_start, a_end), (o_start, o_end), sentiment).
    """
    triplets = []
    for q in quads:
        as_ = q.get("a_start", -1)
        ae = q.get("a_end", -1)
        os_ = q.get("o_start", -1)
        oe = q.get("o_end", -1)
        senti = q.get("sentiment", "INVALID").upper()
        if as_ != -1 and os_ != -1 and as_ < ae and os_ < oe:
            triplets.append(((as_, ae), (os_, oe), senti))
    return triplets
