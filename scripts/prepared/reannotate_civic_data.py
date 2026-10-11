"""Re-annotate civic commentary candidates using Gemini with exact token span grounding,
specifically enforcing multi-quad extraction and 23-category schema resolution.
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

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acos.annotate_and_ground import (
    CATEGORIES_23,
    SENTIMENTS,
    find_token_span,
    ground_quads,
    load_gemini_api_key,
)

WORKING_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
]


def build_civic_prompt(batch: List[Tuple[str, str]]) -> str:
    categories_str = "\n".join(f"- {c}" for c in sorted(CATEGORIES_23))

    examples_str = """
Example 1 (Contrasting Civic Sentiments on Education & Facilities):
Sentence: "የትምህርት ቤቱ መምህራን ብቁ ናቸው ነገር ግን የመማሪያ ክፍሎቹ እጥረት ከፍተኛ ችግር ነው"
Output:
{
  "quads": [
    ["መምህራን", "PUBLIC_SERVICES#EDUCATION", "POSITIVE", "ብቁ ናቸው"],
    ["የመማሪያ ክፍሎቹ", "PUBLIC_SERVICES#INFRASTRUCTURE", "NEGATIVE", "እጥረት ከፍተኛ ችግር ነው"],
    ["NULL", "PUBLIC_SERVICES#EDUCATION", "NEGATIVE", "ከፍተኛ ችግር ነው"]
  ]
}

Example 2 (Utility Cost vs Service Quality):
Sentence: "ቴሌዎች እንዴት ናችሁ ዋጋ ቀንሱልን የኑሮ ውድነት እና የናንተን የዳታ ዋጋ ችለን መኖር"
Output:
{
  "quads": [
    ["የዳታ ዋጋ", "ECONOMY#UTILITIES", "NEGATIVE", "ችለን መኖር"],
    ["ዋጋ", "ECONOMY#UTILITIES", "NEGATIVE", "ቀንሱልን"],
    ["ቴሌዎች", "INFRASTRUCTURE#UTILITIES", "NEGATIVE", "ቀንሱልን"]
  ]
}

Example 3 (Water Supply & Outage vs Tariff):
Sentence: "የመጠጥ ውሃ አቅርቦት እጥረት ከፍተኛ ቢሆንም የውሃ ቢል ክፍያው ግን ያለ አግባብ ጨምሯል"
Output:
{
  "quads": [
    ["የመጠጥ ውሃ አቅርቦት", "PUBLIC_SERVICES#UTILITIES", "NEGATIVE", "እጥረት ከፍተኛ ቢሆንም"],
    ["የውሃ ቢል ክፍያው", "ECONOMY#UTILITIES", "NEGATIVE", "ያለ አግባብ ጨምሯል"]
  ]
}

Example 4 (Crime & Police Services):
Sentence: "በዚህ አካባቢ ሌብነት እና ወንጀል ተስፋፍቷል ፖሊስም ፈጣን እርምጃ አይወስድም"
Output:
{
  "quads": [
    ["ሌብነት እና ወንጀል", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ተስፋፍቷል"],
    ["ፖሊስም", "PUBLIC_SAFETY#CRIME_SERVICES", "NEGATIVE", "ፈጣን እርምጃ አይወስድም"]
  ]
}
"""

    sentences_block = "\n".join(f'[{sid}] "{stext}"' for sid, stext in batch)

    prompt = f"""You are an expert Amharic computational linguist annotating Aspect-Category-Opinion-Sentiment (ACOS) quadruples on Amharic civic, public services, and online governance commentary.

For each numbered Amharic sentence below, extract all valid ACOS quadruples:
- Aspect: The explicit target entity or service evaluated (must be an EXACT substring from the sentence). If implicit (not explicitly named in text), use "NULL".
- Category: EXACTLY one of the 23 valid categories listed below.
- Sentiment: EXACTLY one of ["POSITIVE", "NEGATIVE", "NEUTRAL"].
- Opinion: The explicit evaluative word/phrase (must be an EXACT substring from the sentence). If implicit, use "NULL".

CRITICAL INSTRUCTIONS:
1. MANDATORY MULTI-QUAD EXTRACTION:
   - Do NOT stop after extracting a single quadruple if multiple aspects or evaluations exist.
   - Look for secondary aspects, contrasting clauses (conjunctions like ግን, ሆኖም, ነገር ግን, እንዲሁም, በተጨማሪም, ሳይሆን), sub-aspects (e.g. service quality vs tariff/cost), or overarching evaluations.
2. DISTINGUISHING UTILITIES CATEGORIES:
   - ECONOMY#UTILITIES: Costs, bills, prices, fees, tariff hikes for water, electricity, telecom, energy (e.g., የዳታ ዋጋ, የመብራት ቢል, የውሃ ክፍያ, ታሪፍ).
   - PUBLIC_SERVICES#UTILITIES: Delivery, supply, water availability, power cuts, clean drinking water (e.g., የውሃ አቅርቦት, መብራት መቆራረጥ, የመጠጥ ውሃ).
   - INFRASTRUCTURE#UTILITIES: Physical telecom networks, power lines, transformers, pipes, internet grid (e.g., ቴሌ, የኤሌክትሪክ መስመር, ኔትወርክ, የውሃ ቧንቧ).
3. DISTINGUISHING COMMUNITY SUPPORT:
   - GOVERNANCE#COMMUNITY_SUPPORT: Government aid, safety nets, official relief policies, subsidies.
   - PUBLIC_SERVICES#COMMUNITY_SUPPORT: Public community facilities, shelters, youth centers, municipal social centers.
   - SOCIAL#COMMUNITY_SUPPORT: Grassroots mutual aid, neighbors, edir, traditional social cohesion.
4. EXACT SUBSTRING SPANS: Any explicit aspect or opinion MUST be an exact verbatim substring from the sentence text.
5. VALID CATEGORIES (Strictly 23):
{categories_str}

{examples_str}

SENTENCES TO ANNOTATE:
{sentences_block}

Format your response as a valid JSON object mapping each sentence ID (as a string) to an object with "quads":
{{
  "0": {{ "quads": [ ["aspect1", "CATEGORY", "SENTIMENT", "opinion1"], ["aspect2", "CATEGORY", "SENTIMENT", "opinion2"] ] }},
  "1": {{ "quads": [ ["aspect1", "CATEGORY", "SENTIMENT", "opinion1"] ] }}
}}
Return ONLY the raw JSON object, without backticks or markdown formatting.
"""
    return prompt


def call_gemini_pool(prompt: str, api_key: str, retries: int = 3) -> Dict[str, any]:
    for attempt in range(1, retries + 1):
        for m in WORKING_MODELS:
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
                    time.sleep(1.0)
                    continue
                else:
                    time.sleep(1.0)
            except Exception:
                time.sleep(1.0)
        time.sleep(2 * attempt)
    return {}


def reannotate_civic_candidates(
    candidates_file: Path,
    output_file: Path,
    max_samples: int = 2500,
    batch_size: int = 8,
    num_workers: int = 4,
) -> None:
    api_key = load_gemini_api_key()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
            if len(candidates) >= max_samples:
                break

    print(f"[Civic Annotator] Loaded {len(candidates)} candidate sentences.")

    cached_done: Dict[str, dict] = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached_done[rec["text"]] = rec
        print(f"[Civic Annotator] Found {len(cached_done)} already processed sentences in checkpoint.")

    remaining = [c for c in candidates if c["text"] not in cached_done]
    print(f"[Civic Annotator] Sentences remaining to annotate: {len(remaining)}")

    batches = []
    for i in range(0, len(remaining), batch_size):
        chunk = remaining[i : i + batch_size]
        batch_items = [(str(idx), c["text"]) for idx, c in enumerate(chunk)]
        batches.append((batch_items, chunk))

    out_f = open(output_file, "a", encoding="utf-8")
    completed = len(cached_done)

    def process_task(task):
        batch_items, chunk = task
        prompt = build_civic_prompt(batch_items)
        res = call_gemini_pool(prompt, api_key)
        results = []
        for sid, text in batch_items:
            raw_quads = res.get(sid, {}).get("quads", [])
            grounded = ground_quads(text, raw_quads)
            if grounded:
                results.append({
                    "text": text,
                    "tokens": text.split(),
                    "quads": grounded,
                })
        return results

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_task, b): b for b in batches}
        for future in as_completed(futures):
            try:
                res_list = future.result()
                for rec in res_list:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    completed += 1
                print(f"[Civic Progress] Valid annotated records: {completed}/{len(candidates)}", flush=True)
            except Exception as e:
                print(f"Batch failed: {e}", flush=True)

    out_f.close()
    print(f"\n[Civic Annotator] Completed re-annotation! Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-annotate civic commentary candidates")
    parser.add_argument("--candidates", type=str, default="data/prepared_candidates.jsonl")
    parser.add_argument("--output", type=str, default="data/prepared_reannotated.jsonl")
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    reannotate_civic_candidates(
        candidates_file=Path(args.candidates),
        output_file=Path(args.output),
        max_samples=args.limit,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
