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


def canonical_sort_quads(quads: list[dict]) -> list[dict]:
    """
    Sorts quadruples into a deterministic left-to-right canonical order:
      1. Explicit aspects sorted ascending by a_start.
      2. Implicit aspects with explicit opinions sorted ascending by o_start.
      3. Fully implicit quads placed last, sorted by category then sentiment.
    """
    def sort_key(q: dict):
        a_s = q.get("a_start", -1)
        o_s = q.get("o_start", -1)

        has_explicit_a = a_s is not None and a_s >= 0
        has_explicit_o = o_s is not None and o_s >= 0

        if has_explicit_a:
            return (0, a_s, o_s if has_explicit_o else 9999, q.get("category", ""), q.get("sentiment", ""))
        elif has_explicit_o:
            return (1, 9999, o_s, q.get("category", ""), q.get("sentiment", ""))
        else:
            return (2, 9999, 9999, q.get("category", ""), q.get("sentiment", ""))

    return sorted(quads, key=sort_key)


def quads_to_target(quads: list[dict], tokens: list[str], canonical: bool = True) -> str:
    """
    Serializes a list of quadruple annotations into a standardized target string:
      "[ aspect | category | sentiment | opinion ] ; ..."
    When canonical=True, sorts quads left-to-right by source sentence appearance.
    """
    if not quads:
        return "None"

    sorted_quads = canonical_sort_quads(quads) if canonical else quads

    formatted_quads = []
    for quad in sorted_quads:
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
        if strict_categories and cat_clean not in valid_cats:
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


ORDER7_TO_ORDER6 = {
    "\u1209": "\u120d",  # ሉ -> ል
    "\u1211": "\u1215",  # ሙ -> ም
    "\u1221": "\u1225",  # ሡ -> ሥ
    "\u1229": "\u122d",  # ሩ -> ር
    "\u1239": "\u123d",  # ሱ -> ስ
    "\u1241": "\u1245",  # ሹ -> ሽ
    "\u1249": "\u124d",  # ቁ -> ቅ
    "\u1261": "\u1265",  # ቡ -> ብ
    "\u1271": "\u1275",  # ቱ -> ት
    "\u1279": "\u127d",  # ቹ -> ች
    "\u1289": "\u128d",  # ኁ -> ኅ
    "\u1291": "\u1295",  # ኑ -> ን
    "\u1299": "\u129d",  # ኙ -> ኝ
    "\u12a1": "\u12a5",  # አ -> እ
    "\u12a9": "\u12ad",  # ኩ -> ክ
    "\u12b9": "\u12bd",  # ኹ -> ኽ
    "\u12c9": "\u12cd",  # ዉ -> ው
    "\u12d9": "\u12dd",  # ዙ -> ዝ
    "\u12e1": "\u12e5",  # ዡ -> ዥ
    "\u12e9": "\u12ed",  # ዩ -> ይ
    "\u12f1": "\u12f5",  # ዱ -> ድ
    "\u1301": "\u1305",  # ጁ -> ጅ
    "\u1309": "\u130d",  # ጉ -> ግ
    "\u1321": "\u1325",  # ጡ -> ጥ
    "\u1329": "\u132d",  # ጩ -> ጭ
    "\u1331": "\u1335",  # ጱ -> ጵ
    "\u1339": "\u133d",  # ጹ -> ጽ
    "\u1349": "\u134d",  # ፉ -> ፍ
    "\u1351": "\u1355",  # ፑ -> ፕ
}


def normalize_amharic_clitics(text: str) -> str:
    """
    Normalizes common Amharic clitics (prepositional prefixes, conjunctions,
    and definite article 7th-order consonant-vowel mergers) for clitic-invariant span matching.
    """
    if not text or text == "NULL" or text == "None":
        return text

    words = text.split()
    normalized_words = []

    for word in words:
        w = word.strip()
        if len(w) <= 2:
            normalized_words.append(w)
            continue

        # 1. Strip common prepositional prefixes: እንደ, ስለ, ወደ, በ, ለ, ከ, የ
        for p in ("እንደ", "ስለ", "ወደ", "በ", "ለ", "ከ", "የ"):
            if len(w) >= len(p) + 2 and w.startswith(p):
                w = w[len(p):]
                break

        # 2. Strip conjunction / attachment suffixes: -ም (and/also), -ና (and), -ው, -ዋ
        for s in ("ም", "ና", "ው", "ዋ"):
            if len(w) >= len(s) + 2 and w.endswith(s):
                w = w[:-len(s)]
                break

        # 3. Definite article merger: 7th order vowel merger -> 6th order base consonant
        if len(w) >= 3 and w[-1] in ORDER7_TO_ORDER6:
            w = w[:-1] + ORDER7_TO_ORDER6[w[-1]]

        normalized_words.append(w)

    return " ".join(normalized_words)


def compute_clitic_normalized_set_prf(pred_set: list[tuple], gold_set: list[tuple]) -> tuple[int, int, int]:
    """
    Computes TP, FP, FN with Amharic clitic normalization applied to aspect and opinion spans.
    Tuples can be 4-tuple (a, c, s, o) or 2-tuple (a, o) or 3-tuple (a, s, o).
    """
    def norm_tuple(t: tuple) -> tuple:
        if len(t) == 4:
            return (normalize_amharic_clitics(t[0]), t[1], t[2], normalize_amharic_clitics(t[3]))
        elif len(t) == 2:
            return (normalize_amharic_clitics(t[0]), normalize_amharic_clitics(t[1]))
        elif len(t) == 3:
            return (normalize_amharic_clitics(t[0]), t[1], normalize_amharic_clitics(t[2]))
        return t

    norm_preds = [norm_tuple(p) for p in pred_set]
    norm_golds = [norm_tuple(g) for g in gold_set]
    return compute_set_prf(norm_preds, norm_golds)

