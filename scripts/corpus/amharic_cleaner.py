"""Amharic text normalization, cleaning, and quality filtering module.
Follows OSCAR and mC4 quality filtering methodologies tailored for Ethiopic script.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# Ethiopic Unicode ranges
ETHIOPIC_PATTERN = re.compile(
    r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]"
)

# Ethiopic punctuation marks
ETHIOPIC_WORDSPACE = "\u1361"  # ፡
ETHIOPIC_FULLSTOP = "\u1362"   # ።
ETHIOPIC_COMMA = "\u1363"      # ፥
ETHIOPIC_SEMICOLON = "\u1364"  # ፤
ETHIOPIC_COLON = "\u1365"      # ፦
ETHIOPIC_QUESTION = "\u1366"   # ፧

# Known boilerplate phrases from source platforms
BOILERPLATE_PATTERNS = [
    re.compile(r"www\.ethiobookreview\.com", re.IGNORECASE),
    re.compile(r"ethio\s*book\s*review", re.IGNORECASE),
    re.compile(r"allaboutethio\.com", re.IGNORECASE),
    re.compile(r"allaboutethio", re.IGNORECASE),
    re.compile(r"download\s+free\s+amharic\s+books", re.IGNORECASE),
    re.compile(r"download\s+amharic\s+books", re.IGNORECASE),
    re.compile(r"^\s*ገጽ\s*[:\s]*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*page\s*[:\s]*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*[\(\[]?\s*\d+\s*[\)\]]?\s*$"),
]


def normalize_unicode(text: str) -> str:
    """Normalize text to NFC and strip non-printable/control characters."""
    text = unicodedata.normalize("NFC", text)
    # Remove control chars except newline and tab
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in ("\n", "\t") or not cat.startswith("C"):
            cleaned.append(ch)
        elif ch in ("\r", "\x0c"):
            cleaned.append("\n")
        else:
            cleaned.append(" ")
    return "".join(cleaned)


def is_boilerplate(line: str) -> bool:
    """Check if a line matches known site boilerplate or standalone page numbers."""
    trimmed = line.strip()
    if not trimmed:
        return True
    for pat in BOILERPLATE_PATTERNS:
        if pat.search(trimmed):
            return True
    return False


def get_ethiopic_ratio(text: str) -> float:
    """Calculate the fraction of alphabetical characters that belong to Ethiopic script."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    ethiopic_chars = [c for c in alpha_chars if ETHIOPIC_PATTERN.match(c)]
    return len(ethiopic_chars) / len(alpha_chars)


def clean_line(line: str) -> Optional[str]:
    """Clean and normalize an individual text line."""
    # Collapse multiple spaces and tabs
    line = re.sub(r"[ \t]+", " ", line).strip()
    if not line:
        return None
    if is_boilerplate(line):
        return None
    return line


def clean_document(
    raw_text: str,
    min_doc_chars: int = 100,
    min_ethiopic_ratio: float = 0.50,
    normalize_wordspace: bool = False,
) -> Tuple[Optional[str], Dict[str, any]]:
    """Clean an entire document and return the cleaned text and quality metrics.

    Args:
        raw_text: Raw text extracted from PDF or web page.
        min_doc_chars: Minimum character count required.
        min_ethiopic_ratio: Minimum ratio of Ethiopic characters among alphabetic characters.
        normalize_wordspace: If True, converts Ethiopic word space (፡) to standard space.

    Returns:
        (cleaned_text, quality_meta)
    """
    text = normalize_unicode(raw_text)
    lines = text.split("\n")

    cleaned_lines: List[str] = []
    seen_consecutive: Optional[str] = None

    for raw_line in lines:
        cleaned = clean_line(raw_line)
        if not cleaned:
            continue
        # Deduplicate immediate consecutive repeated lines (e.g. repeated page headers)
        if cleaned == seen_consecutive:
            continue
        seen_consecutive = cleaned
        cleaned_lines.append(cleaned)

    # Join lines into paragraphs
    document = "\n".join(cleaned_lines)
    # Collapse 3+ newlines to 2 newlines
    document = re.sub(r"\n{3,}", "\n\n", document).strip()

    if normalize_wordspace:
        # Optionally replace Ethiopic word separator with standard space
        document = document.replace(ETHIOPIC_WORDSPACE, " ")
        document = re.sub(r"[ ]{2,}", " ", document)

    total_chars = len(document)
    # Ethiopic word characters vs non-Ethiopic alphabetic characters
    ethiopic_chars = len(ETHIOPIC_PATTERN.findall(document))
    non_ws_chars = sum(1 for c in document if not c.isspace())
    alpha_chars = sum(1 for c in document if c.isalpha())
    ethiopic_alphas = sum(1 for c in document if c.isalpha() and ETHIOPIC_PATTERN.match(c))
    ethiopic_ratio = (ethiopic_alphas / alpha_chars) if alpha_chars > 0 else 0.0
    ethiopic_ratio = min(1.0, max(0.0, ethiopic_ratio))

    words = document.split()
    word_count = len(words)

    warnings: List[str] = []
    if total_chars < min_doc_chars:
        warnings.append("short_document")
    if alpha_chars > 0 and ethiopic_ratio < min_ethiopic_ratio:
        warnings.append("low_ethiopic_ratio")

    meta = {
        "char_count": total_chars,
        "word_count": word_count,
        "ethiopic_char_count": ethiopic_chars,
        "ethiopic_ratio": round(ethiopic_ratio, 4),
        "quality_warnings": warnings,
        "is_valid": len(warnings) == 0,
    }

    if not meta["is_valid"]:
        return None, meta

    return document, meta
