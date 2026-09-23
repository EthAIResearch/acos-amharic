"""
Evaluation Metrics for Span-ASTE
================================
Computes standard Precision, Recall, and F1 across four evaluation dimensions:
  1. ASTE: Aspect Sentiment Triplet Extraction (a_span, o_span, sentiment)
  2. AOPE: Aspect-Opinion Pair Extraction (a_span, o_span)
  3. ATE: Aspect Term Extraction (a_span)
  4. OTE: Opinion Term Extraction (o_span)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from span_utils import ID2RELATION, extract_explicit_triplets


def compute_prf(tp: int, fp: int, fn: int) -> dict:
    """Computes precision, recall, f1 safely."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_batch_predictions(
    batch_outputs: list[dict],
    raw_quads: list[list[dict]],
) -> dict:
    """
    Evaluates a list of batch model outputs against raw gold quads.
    Returns aggregated counts for ASTE, AOPE, ATE, OTE, and candidate ceiling.
    """
    counts = {
        "aste": {"tp": 0, "fp": 0, "fn": 0},
        "aope": {"tp": 0, "fp": 0, "fn": 0},
        "ate": {"tp": 0, "fp": 0, "fn": 0},
        "ote": {"tp": 0, "fp": 0, "fn": 0},
        "candidate_ceiling": {"reachable": 0, "total_gold_pairs": 0},
    }

    for b, out in enumerate(batch_outputs):
        quads = raw_quads[b]
        gold_triplets = set(extract_explicit_triplets(quads))
        gold_pairs = {(a, o) for a, o, s in gold_triplets}
        gold_aspects = {a for a, o, s in gold_triplets}
        gold_opinions = {o for a, o, s in gold_triplets}

        # Model predicted triplets: list of (t_span, o_span, rel_id)
        raw_preds = out.get("pred_triplets", [])
        pred_triplets = set()
        for t_span, o_span, rel_id in raw_preds:
            senti = ID2RELATION.get(rel_id, "INVALID")
            if senti != "INVALID":
                pred_triplets.add((t_span, o_span, senti))

        pred_pairs = {(t, o) for t, o, s in pred_triplets}
        pred_aspects = {t for t, o, s in pred_triplets}
        pred_opinions = {o for t, o, s in pred_triplets}

        # 1. ASTE Triplets
        tp_aste = len(pred_triplets & gold_triplets)
        fp_aste = len(pred_triplets - gold_triplets)
        fn_aste = len(gold_triplets - pred_triplets)
        counts["aste"]["tp"] += tp_aste
        counts["aste"]["fp"] += fp_aste
        counts["aste"]["fn"] += fn_aste

        # 2. AOPE Pairs (direct comparison to SDRN)
        tp_aope = len(pred_pairs & gold_pairs)
        fp_aope = len(pred_pairs - gold_pairs)
        fn_aope = len(gold_pairs - pred_pairs)
        counts["aope"]["tp"] += tp_aope
        counts["aope"]["fp"] += fp_aope
        counts["aope"]["fn"] += fn_aope

        # 3. ATE Aspects
        tp_ate = len(pred_aspects & gold_aspects)
        fp_ate = len(pred_aspects - gold_aspects)
        fn_ate = len(gold_aspects - pred_aspects)
        counts["ate"]["tp"] += tp_ate
        counts["ate"]["fp"] += fp_ate
        counts["ate"]["fn"] += fn_ate

        # 4. OTE Opinions
        tp_ote = len(pred_opinions & gold_opinions)
        fp_ote = len(pred_opinions - gold_opinions)
        fn_ote = len(gold_opinions - pred_opinions)
        counts["ote"]["tp"] += tp_ote
        counts["ote"]["fp"] += fp_ote
        counts["ote"]["fn"] += fn_ote

        # 5. Candidate Ceiling (reachability in pruned candidate pools)
        pruned_t = set(out.get("target_candidates", []))
        pruned_o = set(out.get("opinion_candidates", []))
        for a, o in gold_pairs:
            counts["candidate_ceiling"]["total_gold_pairs"] += 1
            if a in pruned_t and o in pruned_o:
                counts["candidate_ceiling"]["reachable"] += 1

    return counts


def summarize_metrics(counts: dict) -> dict:
    """Computes final PRF dictionary from accumulated counts."""
    aste_prf = compute_prf(counts["aste"]["tp"], counts["aste"]["fp"], counts["aste"]["fn"])
    aope_prf = compute_prf(counts["aope"]["tp"], counts["aope"]["fp"], counts["aope"]["fn"])
    ate_prf = compute_prf(counts["ate"]["tp"], counts["ate"]["fp"], counts["ate"]["fn"])
    ote_prf = compute_prf(counts["ote"]["tp"], counts["ote"]["fp"], counts["ote"]["fn"])

    tot_pairs = counts["candidate_ceiling"]["total_gold_pairs"]
    reach_pairs = counts["candidate_ceiling"]["reachable"]
    ceil_recall = reach_pairs / tot_pairs if tot_pairs > 0 else 0.0
    conv_eff = aope_prf["tp"] / reach_pairs if reach_pairs > 0 else 0.0

    return {
        "aste": aste_prf,
        "aope": aope_prf,
        "ate": ate_prf,
        "ote": ote_prf,
        "candidate_ceiling_recall": ceil_recall,
        "candidate_conversion_efficiency": conv_eff,
        "reachable_pairs": reach_pairs,
        "total_gold_pairs": tot_pairs,
    }


def sweep_relation_thresholds(
    candidates_with_scores_per_ex: list[list[tuple[tuple[int, int], tuple[int, int], int, float]]],
    raw_quads_per_ex: list[list[dict]],
    thresholds: list[float] | None = None,
) -> dict:
    """
    Evaluates pair extraction (AOPE) and triplet extraction (ASTE) across a range of
    relation probability thresholds without recomputing model representations.

    candidates_with_scores_per_ex: for each example, list of (t_span, o_span, best_senti_id, p_relation)
    raw_quads_per_ex: for each example, list of gold quads
    """
    if thresholds is None:
        thresholds = [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
        ]

    # Pre-extract gold data per example
    gold_triplets_per_ex = []
    gold_pairs_per_ex = []
    total_gold_pairs = 0
    total_reachable_pairs = 0

    for b, quads in enumerate(raw_quads_per_ex):
        g_triplets = set(extract_explicit_triplets(quads))
        g_pairs = {(a, o) for a, o, s in g_triplets}
        gold_triplets_per_ex.append(g_triplets)
        gold_pairs_per_ex.append(g_pairs)
        total_gold_pairs += len(g_pairs)

        # Reachable pairs in this example's candidate pool
        cand_pairs = {(t, o) for t, o, s, p in candidates_with_scores_per_ex[b]}
        total_reachable_pairs += len(g_pairs & cand_pairs)

    ceiling_recall = total_reachable_pairs / total_gold_pairs if total_gold_pairs > 0 else 0.0

    sweep_results = []
    best_aope_f1 = -1.0
    best_aope_threshold = thresholds[0]
    best_aope_metrics = None

    best_aste_f1 = -1.0
    best_aste_threshold = thresholds[0]
    best_aste_metrics = None

    for t in thresholds:
        tp_aope = fp_aope = fn_aope = 0
        tp_aste = fp_aste = fn_aste = 0

        for b in range(len(raw_quads_per_ex)):
            g_triplets = gold_triplets_per_ex[b]
            g_pairs = gold_pairs_per_ex[b]

            pred_triplets = set()
            pred_pairs = set()

            for t_span, o_span, senti_id, score in candidates_with_scores_per_ex[b]:
                if score >= t:
                    senti_str = ID2RELATION.get(senti_id, "INVALID")
                    if senti_str != "INVALID":
                        pred_triplets.add((t_span, o_span, senti_str))
                        pred_pairs.add((t_span, o_span))

            tp_aope += len(pred_pairs & g_pairs)
            fp_aope += len(pred_pairs - g_pairs)
            fn_aope += len(g_pairs - pred_pairs)

            tp_aste += len(pred_triplets & g_triplets)
            fp_aste += len(pred_triplets - g_triplets)
            fn_aste += len(g_triplets - pred_triplets)

        aope_prf = compute_prf(tp_aope, fp_aope, fn_aope)
        aste_prf = compute_prf(tp_aste, fp_aste, fn_aste)
        conversion = tp_aope / total_reachable_pairs if total_reachable_pairs > 0 else 0.0

        res_entry = {
            "threshold": round(t, 2),
            "aope_precision": round(aope_prf["precision"], 4),
            "aope_recall": round(aope_prf["recall"], 4),
            "aope_f1": round(aope_prf["f1"], 4),
            "aste_precision": round(aste_prf["precision"], 4),
            "aste_recall": round(aste_prf["recall"], 4),
            "aste_f1": round(aste_prf["f1"], 4),
            "conversion_efficiency": round(conversion, 4),
            "tp_aope": tp_aope,
            "fp_aope": fp_aope,
            "fn_aope": fn_aope,
        }
        sweep_results.append(res_entry)

        if aope_prf["f1"] > best_aope_f1:
            best_aope_f1 = aope_prf["f1"]
            best_aope_threshold = t
            best_aope_metrics = res_entry

        if aste_prf["f1"] > best_aste_f1:
            best_aste_f1 = aste_prf["f1"]
            best_aste_threshold = t
            best_aste_metrics = res_entry

    return {
        "candidate_ceiling_recall": round(ceiling_recall, 4),
        "reachable_pairs": total_reachable_pairs,
        "total_gold_pairs": total_gold_pairs,
        "best_aope_threshold": round(best_aope_threshold, 2),
        "best_aope_f1": round(best_aope_f1, 4),
        "best_aope_metrics": best_aope_metrics,
        "best_aste_threshold": round(best_aste_threshold, 2),
        "best_aste_f1": round(best_aste_f1, 4),
        "best_aste_metrics": best_aste_metrics,
        "threshold_sweep": sweep_results,
    }
