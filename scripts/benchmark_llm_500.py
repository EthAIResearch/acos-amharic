"""
Zero-Shot LLM Benchmark on Test Split (Default: 500 Samples)
============================================================
Calls an external LLM (Gemini 1.5 Flash / Pro or GPT-4o-mini) to extract
ACOS quadruples from the first N sentences of data/prepared/test.jsonl,
and evaluates strict, clitic-normalized, and component metrics.

Usage:
    python3 scripts/benchmark_llm_500.py --limit 500
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_byt5_acos"))
from linearization import (
    CATEGORIES,
    SENTIMENTS,
    calculate_metrics,
    compute_clitic_normalized_set_prf,
    compute_set_prf,
    extract_gold_quad_tuples,
)

SYSTEM_PROMPT = f"""You are an expert Amharic NLP linguist performing Aspect-Category-Opinion-Sentiment (ACOS) quadruple extraction.
Given a list of numbered Amharic sentences, extract all ACOS quadruples for each sentence.

Definitions:
- Aspect: The explicit entity, service, or topic being evaluated. If not explicitly mentioned in the text, use "NULL".
- Opinion: The explicit phrase expressing sentiment. If not explicitly mentioned, use "NULL".
- Sentiment: Exactly one of: {sorted(list(SENTIMENTS))}
- Category: Exactly one of the following 22 valid categories:
{sorted(list(CATEGORIES))}

Format your response as a valid JSON object mapping each sentence ID (as a string) to a list of quadruples:
{{
  "0": [["aspect", "CATEGORY", "SENTIMENT", "opinion"]],
  "1": [["NULL", "CATEGORY", "SENTIMENT", "opinion"]]
}}
If a sentence has no quadruple, return an empty list: "2": []
Return ONLY the raw JSON object, without backticks or markdown formatting.
"""


def load_api_key():
    """Loads API key from environment or ~/.env safely."""
    for key_name in ["GEMINI_API_KEY", "OPENAI_API_KEY"]:
        if os.environ.get(key_name):
            return key_name, os.environ[key_name]
    env_path = os.path.expanduser("~/.env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k in ("GEMINI_API_KEY", "OPENAI_API_KEY") and v:
                    return k, v
    return None, None


GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]



def call_gemini(batch_sentences: dict[str, str], api_key: str, model: str | None = None) -> dict[str, list]:
    models_to_try = [model] if model else GEMINI_MODELS
    last_err = None
    for m in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        user_prompt = "Annotate the following Amharic sentences:\n" + json.dumps(batch_sentences, ensure_ascii=False, indent=2)
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_prompt}]}
            ],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
        }
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 404:
            continue
        if resp.status_code == 403:
            raise PermissionError(f"403 PERMISSION_DENIED: {resp.text}")
        if resp.status_code == 429:
            last_err = resp.text
            resp.raise_for_status()
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n", "", text)
            text = re.sub(r"\n```$", "", text)
        return json.loads(text)
    raise RuntimeError(f"All Gemini models returned 404/error. Last error: {last_err}")



def call_openai(batch_sentences: dict[str, str], api_key: str, model: str = "gpt-4o-mini") -> dict[str, list]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user_prompt = "Annotate the following Amharic sentences:\n" + json.dumps(batch_sentences, ensure_ascii=False, indent=2)
    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmark on test split.")
    parser.add_argument("--limit", type=int, default=500, help="Number of test samples to benchmark (default: 500)")
    parser.add_argument("--batch_size", type=int, default=15, help="Sentences per batch (default: 15)")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent worker threads (default: 1 for free tier)")
    parser.add_argument("--delay", type=float, default=4.0, help="Delay between requests in seconds to respect 15 RPM limit (default: 4.0)")
    parser.add_argument("--test_path", default="data/prepared/test.jsonl", help="Path to test.jsonl")
    parser.add_argument("--output", default="results/llm_benchmark_500.json", help="Path to save output results")
    args = parser.parse_args()

    key_type, api_key = load_api_key()
    if not api_key:
        print("\n" + "=" * 80)
        print("ERROR: No API key found (GEMINI_API_KEY or OPENAI_API_KEY).")
        print("To run this benchmark, add your key to ~/.env using the following safe command:")
        print("  printf \"Enter GEMINI_API_KEY (typing hidden): \" && read -s val && echo && echo \"GEMINI_API_KEY=$val\" >> ~/.env && echo \"Saved.\"")
        print("=" * 80 + "\n")
        sys.exit(1)

    print(f"Using API provider: {key_type}")
    print(f"Loading first {args.limit} samples from {args.test_path}...")
    samples = []
    with open(args.test_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.limit:
                break
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples. Chunking into batches of {args.batch_size}...")
    batches = []
    for i in range(0, len(samples), args.batch_size):
        batch = {str(j): samples[j]["text"] for j in range(i, min(i + args.batch_size, len(samples)))}
        batches.append(batch)

    predictions = {}
    total_batches = len(batches)
    print(f"Executing {total_batches} batches (concurrency={args.concurrency}, delay={args.delay}s)...")
    start_time = time.time()

    def process_batch_indexed(item):
        batch_idx, b = item
        for attempt in range(6):
            try:
                t0 = time.time()
                if key_type == "GEMINI_API_KEY":
                    res = call_gemini(b, api_key)
                else:
                    res = call_openai(b, api_key)
                elapsed_b = time.time() - t0
                print(f"  [Batch {batch_idx + 1}/{total_batches}] Processed {len(b)} sentences in {elapsed_b:.1f}s.")
                if args.delay > 0:
                    time.sleep(args.delay)
                return res
            except PermissionError as pe:
                print(f"\n[FATAL AUTH ERROR] {pe}")
                print("Your Gemini API key was rejected with PERMISSION_DENIED.")
                print("Please create a fresh key at: https://aistudio.google.com/app/apikey")
                os._exit(1)
            except requests.exceptions.HTTPError as he:
                status_code = getattr(he.response, "status_code", None)
                if status_code == 429:
                    wait_time = (attempt + 1) * 10
                    print(f"  [Rate Limit 429] Batch {batch_idx + 1}: hit free tier 15 RPM quota. Waiting {wait_time}s before retry ({attempt + 1}/6)...")
                    time.sleep(wait_time)
                else:
                    time.sleep(3)
            except Exception as e:
                if "429" in str(e):
                    wait_time = (attempt + 1) * 10
                    print(f"  [Rate Limit 429] Batch {batch_idx + 1}: hit quota. Waiting {wait_time}s before retry ({attempt + 1}/6)...")
                    time.sleep(wait_time)
                elif attempt == 5:
                    print(f"[ERROR] Batch {batch_idx + 1} failed after 6 attempts: {e}")
                    return {k: [] for k in b}
                else:
                    time.sleep(3)

    indexed_batches = list(enumerate(batches))
    if args.concurrency <= 1:
        for item in indexed_batches:
            res = process_batch_indexed(item)
            if isinstance(res, dict):
                predictions.update(res)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            results = executor.map(process_batch_indexed, indexed_batches)
            for res in results:
                if isinstance(res, dict):
                    predictions.update(res)

    elapsed = time.time() - start_time
    print(f"Completed {len(predictions)} sentence annotations in {elapsed:.1f}s ({len(predictions)/elapsed:.1f} sent/sec).")


    # Evaluate against ground truth
    total_strict_tp, total_strict_fp, total_strict_fn = 0, 0, 0
    total_norm_tp, total_norm_fp, total_norm_fn = 0, 0, 0
    cat_sent_tp, cat_sent_fp, cat_sent_fn = 0, 0, 0
    aste_tp, aste_fp, aste_fn = 0, 0, 0

    evaluated_records = []
    for idx, sample in enumerate(samples):
        golds = extract_gold_quad_tuples(sample)
        raw_preds = predictions.get(str(idx), [])
        clean_preds = []
        for q in raw_preds:
            if isinstance(q, (list, tuple)) and len(q) == 4:
                clean_preds.append((str(q[0]).strip(), str(q[1]).strip(), str(q[2]).strip(), str(q[3]).strip()))

        s_tp, s_fp, s_fn = compute_set_prf(clean_preds, golds)
        n_tp, n_fp, n_fn = compute_clitic_normalized_set_prf(clean_preds, golds)

        p_cs = [(c, s) for (_, c, s, _) in clean_preds]
        g_cs = [(c, s) for (_, c, s, _) in golds]
        cs_tp, cs_fp, cs_fn = compute_set_prf(p_cs, g_cs)

        p_as = [(a, s, o) for (a, _, s, o) in clean_preds]
        g_as = [(a, s, o) for (a, _, s, o) in golds]
        as_tp, as_fp, as_fn = compute_set_prf(p_as, g_as)

        total_strict_tp += s_tp
        total_strict_fp += s_fp
        total_strict_fn += s_fn

        total_norm_tp += n_tp
        total_norm_fp += n_fp
        total_norm_fn += n_fn

        cat_sent_tp += cs_tp
        cat_sent_fp += cs_fp
        cat_sent_fn += cs_fn

        aste_tp += as_tp
        aste_fp += as_fp
        aste_fn += as_fn

        evaluated_records.append({
            "id": idx,
            "text": sample["text"],
            "gold": golds,
            "pred": clean_preds,
            "strict_tp": s_tp,
            "norm_tp": n_tp,
        })

    strict_m = calculate_metrics(total_strict_tp, total_strict_fp, total_strict_fn)
    norm_m = calculate_metrics(total_norm_tp, total_norm_fp, total_norm_fn)
    cs_m = calculate_metrics(cat_sent_tp, cat_sent_fp, cat_sent_fn)
    aste_m = calculate_metrics(aste_tp, aste_fp, aste_fn)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary = {
        "samples_evaluated": len(samples),
        "strict_metrics": strict_m,
        "clitic_normalized_metrics": norm_m,
        "category_sentiment_metrics": cs_m,
        "aste_metrics": aste_m,
        "details": evaluated_records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"OFFICIAL BENCHMARK REPORT: LLM ZERO-SHOT ON FIRST {len(samples)} TEST SAMPLES")
    print("=" * 80)
    print(f"1. STRICT 4-WAY FULL QUAD EXACT MATCH:")
    print(f"   Precision: {strict_m['precision']*100:.2f}% | Recall: {strict_m['recall']*100:.2f}% | F1: {strict_m['f1']*100:.2f}%")
    print(f"   (TP: {total_strict_tp}, FP: {total_strict_fp}, FN: {total_strict_fn})")
    print(f"\n2. CLITIC-NORMALIZED FULL QUAD MATCH:")
    print(f"   Precision: {norm_m['precision']*100:.2f}% | Recall: {norm_m['recall']*100:.2f}% | F1: {norm_m['f1']*100:.2f}%")
    print(f"   (TP: {total_norm_tp}, FP: {total_norm_fp}, FN: {total_norm_fn})")
    print(f"\n3. CATEGORY + SENTIMENT ONLY:")
    print(f"   Precision: {cs_m['precision']*100:.2f}% | Recall: {cs_m['recall']*100:.2f}% | F1: {cs_m['f1']*100:.2f}%")
    print(f"   (TP: {cat_sent_tp}, FP: {cat_sent_fp}, FN: {cat_sent_fn})")
    print(f"\n4. ASTE (Aspect + Sentiment + Opinion, ignoring Category):")
    print(f"   Precision: {aste_m['precision']*100:.2f}% | Recall: {aste_m['recall']*100:.2f}% | F1: {aste_m['f1']*100:.2f}%")
    print(f"   (TP: {aste_tp}, FP: {aste_fp}, FN: {aste_fn})")
    print("=" * 80)
    print(f"Detailed record saved to: {args.output}")


if __name__ == "__main__":
    main()
