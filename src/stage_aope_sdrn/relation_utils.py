"""
Pure-Python (no torch) helpers for building the token-pair relation matrix
used by the joint SDRN-style AOPE model (Chen et al., ACL 2020,
"Synchronous Double-channel Recurrent Network for Aspect-Opinion Pair
Extraction"). Kept dependency-free so it's directly unit-testable, same
pattern as bio_labels.py and pair_utils.py elsewhere in this project.
"""


def explicit_pairs(quads: list) -> list:
    """Extract ((a_start, a_end), (o_start, o_end)) tuples for explicit pairs."""
    return [
        ((q["a_start"], q["a_end"]), (q["o_start"], q["o_end"]))
        for q in quads
        if q["a_start"] != -1 and q["o_start"] != -1
    ]


def build_word_relation_matrix(n_tokens: int, pairs: list) -> list:
    """
    pairs: list of ((a_start, a_end), (o_start, o_end)) word-index spans
        for explicit-both aspect-opinion pairs in one sentence.
    Returns an n_tokens x n_tokens binary matrix (list of lists) where
    matrix[i][j] = 1 iff token i is inside some aspect span and token j is
    inside the paired opinion span (or vice versa) -- symmetric, matching
    SDRN's relation matrix Z (Section 3.2.2, Eq. 6-7 of Chen et al. 2020).
    """
    matrix = [[0] * n_tokens for _ in range(n_tokens)]
    for (a1, a2), (o1, o2) in pairs:
        for i in range(a1, a2):
            for j in range(o1, o2):
                matrix[i][j] = 1
                matrix[j][i] = 1
    return matrix


def correlation_degree(rel_matrix, a_span: tuple, o_span: tuple) -> float:
    """
    SDRN's inference-time pair scoring (Eq. 17): the bidirectional average
    relation score between an aspect span and an opinion span, using the
    (predicted or gold) token-pair relation matrix. A pair is accepted if
    this exceeds a threshold (0.5 default, matching the paper).
    """
    a1, a2 = a_span
    o1, o2 = o_span
    len_a, len_o = a2 - a1, o2 - o1
    if len_a == 0 or len_o == 0:
        return 0.0

    sum_ao = sum(rel_matrix[k][l] for k in range(a1, a2) for l in range(o1, o2))
    sum_oa = sum(rel_matrix[l][k] for l in range(o1, o2) for k in range(a1, a2))
    return 0.5 * (sum_ao / (len_a * len_o) + sum_oa / (len_o * len_a))


def compute_prf(pred_items_per_ex: list, gold_items_per_ex: list) -> dict:
    """Standard micro-averaged precision, recall, and F1 for sets of extracted items."""
    tp = fp = fn = 0
    for preds, golds in zip(pred_items_per_ex, gold_items_per_ex):
        p, g = set(preds), set(golds)
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def sweep_thresholds(
    candidates_with_scores_per_ex: list,
    gold_pairs_per_ex: list,
    thresholds: list | None = None,
) -> dict:
    """
    Evaluates pair PRF across a list of correlation degree thresholds without recomputing
    model representations or span predictions.
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    all_candidates_per_ex = [
        [pair for pair, _ in ex_candidates]
        for ex_candidates in candidates_with_scores_per_ex
    ]
    ceiling_prf = compute_prf(all_candidates_per_ex, gold_pairs_per_ex)

    sweep_results = []
    best_f1 = -1.0
    best_threshold = thresholds[0]
    best_metrics = None

    for t in thresholds:
        filtered_pairs_per_ex = [
            [pair for pair, score in ex_candidates if score >= t]
            for ex_candidates in candidates_with_scores_per_ex
        ]
        prf = compute_prf(filtered_pairs_per_ex, gold_pairs_per_ex)
        prf["threshold"] = round(t, 4)
        sweep_results.append(prf)
        if prf["f1"] > best_f1:
            best_f1 = prf["f1"]
            best_threshold = t
            best_metrics = prf

    return {
        "candidate_ceiling_recall": ceiling_prf["recall"],
        "candidate_ceiling_prf": ceiling_prf,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "sweep": sweep_results,
    }

