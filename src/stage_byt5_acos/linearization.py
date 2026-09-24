"""
Target Linearization and Robust Deserialization for ByT5 ACOS Quadruple Extraction
==================================================================================
Formats and parses structured quadruple strings for generative seq2seq models:
  "[ aspect | category | sentiment | opinion ] ; [ aspect | category | sentiment | opinion ]"

Special Tokens:
  - Implicit aspect or opinion is represented as "NULL".
  - Empty target (no quads in sentence) is represented as "None".
"""
import re

CATEGORIES = {
    "ECONOMY#COMMUNITY_SUPPORT",
    "ECONOMY#EMPLOYMENT",
    "ECONOMY#TAXATION",
    "GOVERNANCE#CITIZEN_ENGAGEMENT",
    "GOVERNANCE#COMMUNITY_SUPPORT",
    "GOVERNANCE#LEGISLATION",
    "GOVERNANCE#TRANSPARENCY",
    "HEALTHCARE#GENERAL",
    "INFRASTRUCTURE#TRANSPORTATION",
    "INFRASTRUCTURE#UTILITIES",
    "PUBLIC_SAFETY#CRIME",
    "PUBLIC_SAFETY#CRIME_SERVICES",
    "PUBLIC_SAFETY#EMERGENCY_SERVICES",
    "PUBLIC_SERVICES#COMMUNITY_SUPPORT",
    "PUBLIC_SERVICES#EDUCATION",
    "PUBLIC_SERVICES#EMERGENCY_SERVICES",
    "PUBLIC_SERVICES#HEALTHCARE",
    "PUBLIC_SERVICES#INFRASTRUCTURE",
    "PUBLIC_SERVICES#UTILITIES",
    "SOCIAL#COMMUNITY_SUPPORT",
    "SOCIAL#EDUCATION",
    "SOCIAL#EQUALITY_JUSTICE",
}

SENTIMENTS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}


def quads_to_target(quads: list[dict], tokens: list[str]) -> str:
    """
    Serializes a list of quadruple annotations into a standardized target string:
      "[ aspect | category | sentiment | opinion ] ; ..."
    """
    if not quads:
        return "None"

    formatted_quads = []
    for quad in quads:
        a_s = quad.get("a_start", -1)
        a_e = quad.get("a_end", -1)
        o_s = quad.get("o_start", -1)
        o_e = quad.get("o_end", -1)

        # Aspect
        if a_s is not None and a_s >= 0 and a_e is not None and a_e > a_s and a_e <= len(tokens):
            aspect = " ".join(tokens[a_s:a_e]).strip()
        else:
            aspect = "NULL"

        # Opinion
        if o_s is not None and o_s >= 0 and o_e is not None and o_e > o_s and o_e <= len(tokens):
            opinion = " ".join(tokens[o_s:o_e]).strip()
        else:
            opinion = "NULL"

        category = quad.get("category", "").strip()
        sentiment = quad.get("sentiment", "").strip().upper()

        if not category:
            continue
        if not sentiment:
            sentiment = "NEUTRAL"

        formatted_quads.append(f"[ {aspect} | {category} | {sentiment} | {opinion} ]")

    if not formatted_quads:
        return "None"

    return " ; ".join(formatted_quads)


def parse_target_to_quads(
    target_str: str,
    strict_categories: bool = True,
    valid_categories: set[str] | None = None,
) -> list[tuple[str, str, str, str]]:
    """
    Robustly parses a generated target string into structured quadruple tuples:
      (aspect, category, sentiment, opinion)

    Handles:
      - Bracketed items: [ a | c | s | o ]
      - Semicolon or newline separation
      - Missing brackets or extra whitespace
      - Normalization of "none" / "null"
    """
    if not target_str:
        return []

    target_str = target_str.strip()
    if target_str.lower() in {"none", "none.", "null", ""}:
        return []

    valid_cats = valid_categories or CATEGORIES

    # Extract bracketed blocks e.g. [ ... ]
    bracket_blocks = re.findall(r"\[(.*?)\]", target_str)
    raw_blocks = bracket_blocks if bracket_blocks else target_str.split(";")

    parsed_quads = []
    for block in raw_blocks:
        parts = [p.strip() for p in block.split("|")]
        if len(parts) != 4:
            continue

        aspect, category, sentiment, opinion = parts

        # Normalize sentiment
        sent_upper = sentiment.upper()
        if sent_upper not in SENTIMENTS:
            # Fallback search if sentiment has slight noise
            matched_sent = None
            for s in SENTIMENTS:
                if s in sent_upper:
                    matched_sent = s
                    break
            if not matched_sent:
                continue
            sent_upper = matched_sent

        # Validate category
        cat_clean = category.strip()
        if strict_categories:
            if cat_clean not in valid_cats:
                # Attempt case-insensitive or partial match
                matched_cat = None
                for c in valid_cats:
                    if c.lower() == cat_clean.lower():
                        matched_cat = c
                        break
                if not matched_cat:
                    continue
                cat_clean = matched_cat

        # Normalize NULL markers
        aspect_norm = "NULL" if aspect.upper() in {"NULL", "-1", "NONE", ""} else aspect
        opinion_norm = "NULL" if opinion.upper() in {"NULL", "-1", "NONE", ""} else opinion

        parsed_quads.append((aspect_norm, cat_clean, sent_upper, opinion_norm))

    return parsed_quads


def extract_gold_quad_tuples(sample: dict) -> list[tuple[str, str, str, str]]:
    """
    Extracts gold (aspect, category, sentiment, opinion) tuples directly from a dataset sample.
    """
    tokens = sample.get("tokens", [])
    quads = sample.get("quads", [])
    gold_tuples = []

    for quad in quads:
        a_s = quad.get("a_start", -1)
        a_e = quad.get("a_end", -1)
        o_s = quad.get("o_start", -1)
        o_e = quad.get("o_end", -1)

        aspect = " ".join(tokens[a_s:a_e]).strip() if (a_s >= 0 and a_e > a_s and a_e <= len(tokens)) else "NULL"
        opinion = " ".join(tokens[o_s:o_e]).strip() if (o_s >= 0 and o_e > o_s and o_e <= len(tokens)) else "NULL"
        category = quad.get("category", "").strip()
        sentiment = quad.get("sentiment", "").strip().upper()

        if category and sentiment:
            gold_tuples.append((aspect, category, sentiment, opinion))

    return gold_tuples


def compute_set_prf(pred_set: list[tuple], gold_set: list[tuple]) -> tuple[int, int, int]:
    """Computes TP, FP, FN between predicted and gold multiset of tuples."""
    gold_remaining = list(gold_set)
    tp = 0
    fp = 0

    for pred in pred_set:
        if pred in gold_remaining:
            tp += 1
            gold_remaining.remove(pred)
        else:
            fp += 1
    fn = len(gold_remaining)
    return tp, fp, fn


def calculate_metrics(tp: int, fp: int, fn: int) -> dict:
    """Calculates precision, recall, and F1 given TP, FP, FN counts."""
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
