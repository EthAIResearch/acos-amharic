"""Annotate candidate sentences with Gemini and ground exact token spans for ACOS quadruples
covering all 23 mandated categories, with strong priority on multi-quadruple sentences.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

CATEGORIES_23 = [
    "ECONOMY#COMMUNITY_SUPPORT",
    "ECONOMY#EMPLOYMENT",
    "ECONOMY#TAXATION",
    "ECONOMY#UTILITIES",
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
]

SENTIMENTS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}


def load_gemini_api_key() -> str:
    """Load GEMINI_API_KEY from environment or ~/.env safely."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env_file = Path.home() / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    return line.strip().split("=", 1)[1]
    raise RuntimeError("GEMINI_API_KEY not found in environment or ~/.env")


def build_prompt(batch: List[Tuple[int, str]]) -> str:
    categories_str = "\n".join(f"- {c}" for c in sorted(CATEGORIES_23))
    examples_str = """
Example 1:
Sentence: "የትምህርት ቤቱ መምህራን ብቁ ናቸው ነገር ግን የመማሪያ ክፍሎቹ እጥረት ከፍተኛ ችግር ነው"
Output:
{
  "quads": [
    ["መምህራን", "PUBLIC_SERVICES#EDUCATION", "POSITIVE", "ብቁ ናቸው"],
    ["የመማሪያ ክፍሎቹ", "PUBLIC_SERVICES#INFRASTRUCTURE", "NEGATIVE", "እጥረት ከፍተኛ ችግር ነው"],
    ["NULL", "PUBLIC_SERVICES#EDUCATION", "NEGATIVE", "ከፍተኛ ችግር ነው"]
  ]
}

Example 2:
Sentence: "የከተማው የውሃ አቅርቦት መሻሻል ቢያሳይም የኤሌክትሪክ ታሪፉ ግን በጣም ውድ ነው"
Output:
{
  "quads": [
    ["የውሃ አቅርቦት", "PUBLIC_SERVICES#UTILITIES", "POSITIVE", "መሻሻል ቢያሳይም"],
    ["የኤሌክትሪክ ታሪፉ", "ECONOMY#UTILITIES", "NEGATIVE", "በጣም ውድ ነው"],
    ["NULL", "INFRASTRUCTURE#UTILITIES", "POSITIVE", "መሻሻል ቢያሳይም"]
  ]
}

Example 3:
Sentence: "በዚህ አካባቢ ሌብነት እና ወንጀል ተስፋፍቷል ፖሊስም ፈጣን እርምጃ አይወስድም ፍርድ ቤቶችም ችላ ብለውታል"
Output:
{
  "quads": [
    ["ሌብነት እና ወንጀል", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ተስፋፍቷል"],
    ["ፖሊስም", "PUBLIC_SAFETY#CRIME_SERVICES", "NEGATIVE", "ፈጣን እርምጃ አይወስድም"],
    ["ፍርድ ቤቶችም", "PUBLIC_SAFETY#CRIME_SERVICES", "NEGATIVE", "ችላ ብለውታል"]
  ]
}

Example 4:
Sentence: "የስራ እድል ማጣት ለወጣቱ ፈተና ቢሆንም የማህበረሰቡ መረዳዳት ግን ከፍተኛ ጥንካሬ ሆኖታል"
Output:
{
  "quads": [
    ["የስራ እድል", "ECONOMY#EMPLOYMENT", "NEGATIVE", "ማጣት ለወጣቱ ፈተና ቢሆንም"],
    ["የማህበረሰቡ መረዳዳት", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "ከፍተኛ ጥንካሬ ሆኖታል"]
  ]
}
"""

    sentences_block = "\n".join(f'[{sid}] "{stext}"' for sid, stext in batch)

    prompt = f"""You are an expert Amharic computational linguist annotating Aspect-Category-Opinion-Sentiment (ACOS) quadruples.

For each numbered Amharic sentence below, extract all valid ACOS quadruples:
- Aspect: The explicit target entity or service evaluated (must be an EXACT substring from the sentence). If implicit (not explicitly named in text), use "NULL".
- Category: EXACTLY one of the 23 valid categories listed below.
- Sentiment: EXACTLY one of ["POSITIVE", "NEGATIVE", "NEUTRAL"].
- Opinion: The explicit evaluative word/phrase (must be an EXACT substring from the sentence). If implicit, use "NULL".

CRITICAL INSTRUCTIONS:
1. MANDATORY MULTI-QUAD PREVALENCE: The user explicitly requires that MOST sentences have MULTIPLE quadruples (typically 2 to 4 quads).
   - DO NOT stop after extracting a single quadruple.
   - Look for secondary aspects, contrasting clauses (conjunctions like ግን, ሆኖም, ነገር ግን, እንዲሁም), sub-aspects (e.g. quality vs cost, service vs facilities), or implicit overarching evaluations.
   - If a sentence evaluates an entity, identify both the specific aspect and any secondary or implicit aspects present.
2. CATEGORY TAXONOMY: You must strictly choose from these 23 categories:
{categories_str}
3. EXACT SUBSTRING SPANS: Any explicit aspect or opinion MUST be an exact verbatim substring from the sentence.
4. EMPTY EVALUATIONS: Only return [] if the sentence is completely neutral narrative with zero evaluative opinion.

{examples_str}

SENTENCES TO ANNOTATE:
{sentences_block}

Format your response as a valid JSON object mapping each sentence ID (as a string) to an object with "quads":
{{
  "0": {{ "quads": [ ["aspect", "CATEGORY", "SENTIMENT", "opinion"], ["aspect2", "CATEGORY", "SENTIMENT", "opinion2"] ] }},
  "1": {{ "quads": [ ["aspect", "CATEGORY", "SENTIMENT", "opinion"], ["aspect2", "CATEGORY", "SENTIMENT", "opinion2"] ] }}
}}
Return ONLY the raw JSON object, without backticks or markdown formatting.
"""
    return prompt


WORKING_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
]
_model_idx = 0


def call_gemini(prompt: str, api_key: str, model: Optional[str] = None, retries: int = 3) -> Dict[str, any]:
    global _model_idx
    models_to_try = [model] if model else WORKING_MODELS

    for attempt in range(1, retries + 1):
        for m in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
            }
            try:
                resp = requests.post(url, json=payload, timeout=45)
                if resp.status_code == 200:
                    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
                    text = re.sub(r"\s*```$", "", text)
                    return json.loads(text)
                elif resp.status_code == 429:
                    # Model rate-limited or quota exceeded, try next model in pool
                    time.sleep(1.0)
                    continue
                else:
                    time.sleep(1.0)
            except Exception:
                time.sleep(1.0)
        time.sleep(2 * attempt)
    return {}


def find_token_span(tokens: List[str], phrase: str, sentence_text: str) -> Tuple[int, int]:
    """Ground a phrase substring to exact token start and end indices."""
    if not phrase or phrase.strip() == "NULL":
        return -1, -1
    phrase = phrase.strip()
    if not phrase:
        return -1, -1

    pw = phrase.split()
    L = len(pw)

    # 1. Exact token matching
    for i in range(len(tokens) - L + 1):
        if tokens[i : i + L] == pw:
            return i, i + L

    # 2. Punctuation-stripped token matching
    clean_target = [w.strip('.,!?:;፤፡፦"\'()') for w in pw]
    for i in range(len(tokens) - L + 1):
        clean_slice = [t.strip('.,!?:;፤፡፦"\'()') for t in tokens[i : i + L]]
        if clean_slice == clean_target:
            return i, i + L

    # 3. Substring character-level search in original sentence text
    idx = sentence_text.find(phrase)
    if idx != -1:
        end_idx = idx + len(phrase)
        cur = 0
        s_tok, e_tok = None, None
        for ti, tok in enumerate(tokens):
            t_start = sentence_text.find(tok, cur)
            if t_start == -1:
                t_start = cur
            t_end = t_start + len(tok)
            cur = t_end
            if s_tok is None and (t_start <= idx < t_end or idx <= t_start < end_idx):
                s_tok = ti
            if s_tok is not None and t_end >= end_idx:
                e_tok = ti + 1
                break
        if s_tok is not None and e_tok is not None and e_tok > s_tok:
            return s_tok, e_tok

    return -1, -1


def ground_quads(
    sentence_text: str,
    raw_quads: List[List[str]],
) -> List[Dict[str, any]]:
    """Ground raw quadruples to validated token spans."""
    tokens = sentence_text.split()
    grounded = []

    for item in raw_quads:
        if len(item) != 4:
            continue
        asp_str, cat_str, sent_str, op_str = item

        cat = cat_str.strip()
        if cat not in CATEGORIES_23:
            # Check for close matches or whitespace anomalies
            cat = cat.replace(" ", "")
            if cat not in CATEGORIES_23:
                continue

        sent = sent_str.strip().upper()
        if sent not in SENTIMENTS:
            continue

        a_start, a_end = find_token_span(tokens, asp_str, sentence_text)
        o_start, o_end = find_token_span(tokens, op_str, sentence_text)

        # Discard invalid spans where non-NULL could not be grounded
        if asp_str.strip() != "NULL" and (a_start == -1 or a_end == -1):
            continue
        if op_str.strip() != "NULL" and (o_start == -1 or o_end == -1):
            continue

        grounded.append({
            "a_start": a_start,
            "a_end": a_end,
            "category": cat,
            "sentiment": sent,
            "o_start": o_start,
            "o_end": o_end,
        })

    return grounded


def run_annotation_pipeline(
    candidates_file: Path,
    output_file: Path,
    max_samples: int = 2500,
    batch_size: int = 8,
    num_workers: int = 4,
) -> None:
    api_key = load_gemini_api_key()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load candidates
    candidates = []
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))
            if len(candidates) >= max_samples:
                break

    print(f"[Annotator] Loaded {len(candidates)} candidate sentences.")

    # Check for existing checkpoint
    cached_grounded: Dict[str, Dict[str, any]] = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cached_grounded[rec["text"]] = rec
        print(f"[Annotator] Found {len(cached_grounded)} already processed sentences in checkpoint.")

    remaining = [c for c in candidates if c["text"] not in cached_grounded]
    print(f"[Annotator] Sentences remaining to annotate: {len(remaining)}")

    batches = []
    for i in range(0, len(remaining), batch_size):
        chunk = remaining[i : i + batch_size]
        batch_items = [(str(idx), c["text"]) for idx, c in enumerate(chunk)]
        batches.append((batch_items, chunk))

    completed_count = len(cached_grounded)
    out_f = open(output_file, "a", encoding="utf-8")

    def process_batch(task):
        batch_items, chunk_candidates = task
        prompt = build_prompt(batch_items)
        res = call_gemini(prompt, api_key)
        results = []
        for sid, text in batch_items:
            quads_raw = res.get(sid, {}).get("quads", [])
            grounded = ground_quads(text, quads_raw)
            if grounded:
                results.append({
                    "text": text,
                    "tokens": text.split(),
                    "quads": grounded,
                })
        return results

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        for future in as_completed(futures):
            try:
                res_list = future.result()
                for rec in res_list:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    completed_count += 1
                print(f"[Progress] Valid annotated sentences: {completed_count}/{len(candidates)}", flush=True)
            except Exception as e:
                print(f"Batch failed: {e}", flush=True)

    out_f.close()
    print(f"\n[Annotator] Annotation complete! Grounded dataset in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate and ground ACOS quadruples")
    parser.add_argument("--candidates", type=str, default="data/book_acos/raw_candidates.jsonl")
    parser.add_argument("--output", type=str, default="data/book_acos/annotated_grounded.jsonl")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_annotation_pipeline(
        candidates_file=Path(args.candidates),
        output_file=Path(args.output),
        max_samples=args.limit,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
