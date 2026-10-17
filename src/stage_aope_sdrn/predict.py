"""
Interactive and batch inference script for Joint AOPE (SDRN).

Extracts aspect terms, opinion terms, and paired relations from Amharic sentences
using a trained Afro-XLM-R Word-Level SDRN model.

Usage Examples:
    # 1. Run inference on the 20 curated example sentences:
    python3 src/stage_aope_sdrn/predict.py \
        --config configs/stage_aope_sdrn_gemini.yaml \
        --checkpoint results/stage_aope_sdrn/afroxlmr_gemini_run1/best_model.pt

    # 2. Test a single sentence:
    python3 src/stage_aope_sdrn/predict.py \
        --checkpoint results/stage_aope_sdrn/afroxlmr_gemini_run1/best_model.pt \
        --text "የኢንተርኔት ፍጥነቱ በጣም ፈጣን ነው ግን ዋጋው ውድ ነው"

    # 3. Interactive prompt mode:
    python3 src/stage_aope_sdrn/predict.py \
        --checkpoint results/stage_aope_sdrn/afroxlmr_gemini_run1/best_model.pt \
        --interactive

    # 4. Run on a custom JSONL or text file:
    python3 src/stage_aope_sdrn/predict.py \
        --checkpoint results/stage_aope_sdrn/afroxlmr_gemini_run1/best_model.pt \
        --file data/example_test_sentences.jsonl
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from align import ID2LABEL_5WAY
from bio_labels import decode_5way_bio_spans
from relation_utils import correlation_degree

try:
    import torch
    from transformers import AutoTokenizer
    from model import JointAOPESDRN
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    AutoTokenizer = None
    JointAOPESDRN = None
    TORCH_AVAILABLE = False


def load_model_and_tokenizer(
    model_name: str,
    checkpoint_path: str,
    use_crf: bool = True,
    num_recurrent_steps: int = 2,
    relation_threshold: float = 0.1,
    device=None,
):
    if device is None:
        device = torch.device("cpu")
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Initializing JointAOPESDRN ({model_name})...")
    model = JointAOPESDRN(
        model_name=model_name,
        num_recurrent_steps=num_recurrent_steps,
        relation_threshold=relation_threshold,
        use_crf=use_crf,
    )

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found! Running with base weights (demo/untrained).")

    model.to(device)
    model.eval()
    return model, tokenizer


def predict_sentence(
    model,
    tokenizer,
    text: str,
    device,
    threshold: float = 0.5,
    word_level: bool = True,
    max_length: int = 256,
    max_words: int = 128,
):
    tokens = text.strip().split()
    n = len(tokens)
    if n == 0:
        return {"text": text, "tokens": [], "aspects": [], "opinions": [], "pairs": []}

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Subword mapping for word pooling
    word_ids_raw = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    ).word_ids()

    L = max_length
    M = max_words
    word_to_subwords = [[] for _ in range(M)]
    for si, wi in enumerate(word_ids_raw):
        if wi is not None and wi < M:
            word_to_subwords[wi].append(si)

    subword_to_word = [[0.0] * L for _ in range(M)]
    word_mask = [False] * M
    for wi in range(min(n, M)):
        sis = word_to_subwords[wi]
        if sis:
            word_mask[wi] = True
            inv_len = 1.0 / len(sis)
            for si in sis:
                subword_to_word[wi][si] = inv_len

    subword_to_word_tensor = torch.tensor([subword_to_word], dtype=torch.float32, device=device)
    word_mask_tensor = torch.tensor([word_mask], dtype=torch.bool, device=device)
    content_mask = torch.tensor(
        [[wid is not None for wid in word_ids_raw] + [False] * (L - len(word_ids_raw))],
        dtype=torch.bool,
        device=device,
    )

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            content_mask=content_mask,
            subword_to_word=subword_to_word_tensor if word_level else None,
            word_mask=word_mask_tensor if word_level else None,
        )

        decode_mask = word_mask_tensor if word_level else content_mask
        pred_ids = model.decode_tags(out["logits"], mask=decode_mask)[0]
        rel_scores = out["relation_logits"][0].cpu().tolist()

    word_tags = [ID2LABEL_5WAY.get(pid, "O") for pid in pred_ids[:n]]
    a_spans, o_spans = decode_5way_bio_spans(word_tags)
    word_rel = [row[:n] for row in rel_scores[:n]]

    aspects = [{"span": (s, e), "text": " ".join(tokens[s:e])} for s, e in a_spans]
    opinions = [{"span": (s, e), "text": " ".join(tokens[s:e])} for s, e in o_spans]

    pairs = []
    for a_span in a_spans:
        for o_span in o_spans:
            score = correlation_degree(word_rel, a_span, o_span)
            if score >= threshold:
                pairs.append(
                    {
                        "aspect": " ".join(tokens[a_span[0] : a_span[1]]),
                        "opinion": " ".join(tokens[o_span[0] : o_span[1]]),
                        "score": round(score, 4),
                        "a_span": a_span,
                        "o_span": o_span,
                    }
                )

    return {
        "text": text,
        "tokens": tokens,
        "aspects": aspects,
        "opinions": opinions,
        "pairs": pairs,
    }


def print_prediction_card(pred: dict, expected: dict = None, index: int = None):
    header = f"=== Example {index} ===" if index is not None else "=== Prediction ==="
    print("\n" + "=" * 65)
    print(header)
    print("=" * 65)
    print(f"Sentence: \"{pred['text']}\"")
    if expected and "domain" in expected:
        print(f"Domain:   [{expected['domain']}]")

    print("\nExtracted Entities:")
    if pred["aspects"]:
        for a in pred["aspects"]:
            print(f"  • Aspect:  \"{a['text']}\" (span: {a['span']})")
    else:
        print("  • Aspect:  (None detected)")

    if pred["opinions"]:
        for o in pred["opinions"]:
            print(f"  • Opinion: \"{o['text']}\" (span: {o['span']})")
    else:
        print("  • Opinion: (None detected)")

    print("\nPredicted Aspect-Opinion Pairs (AOPE):")
    if pred["pairs"]:
        for p in pred["pairs"]:
            print(f"  🔗 [Score: {p['score']:.3f}] \"{p['aspect']}\"  <--->  \"{p['opinion']}\"")
    else:
        print("  (No pairs exceeded the acceptance threshold)")

    if expected:
        print("\nGround Truth Reference:")
        if "expected_aspects" in expected:
            print(f"  ✓ Expected Aspects:  {expected['expected_aspects']}")
        if "expected_opinions" in expected:
            print(f"  ✓ Expected Opinions: {expected['expected_opinions']}")


def run_interactive(model, tokenizer, device, threshold, word_level):
    print("\n" + "=" * 70)
    print(" SDRN Amharic Aspect-Opinion Pair Extraction (Interactive Mode)")
    print(" Enter an Amharic review to extract aspects, opinions, and pairs.")
    print(" Type 'exit' or 'q' to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            line = input("አማርኛ ጽሑፍ ያስገቡ (Enter text) > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                print("Exiting interactive mode. Goodbye!")
                break
            pred = predict_sentence(
                model=model,
                tokenizer=tokenizer,
                text=line,
                device=device,
                threshold=threshold,
                word_level=word_level,
            )
            print_prediction_card(pred)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break


def main():
    parser = argparse.ArgumentParser(description="Test and inference runner for Amharic SDRN AOPE.")
    parser.add_argument("--config", default="configs/stage_aope_sdrn_gemini.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Path to best_model.pt checkpoint")
    parser.add_argument("--text", default=None, help="Single Amharic sentence to test")
    parser.add_argument("--file", default=None, help="Path to input test file (JSONL or TXT)")
    parser.add_argument("--threshold", type=float, default=None, help="Pair acceptance correlation threshold (delta-hat)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal mode")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("\n[Error] PyTorch and Transformers are required to run inference.")
        print("Please run this command inside your environment/container where PyTorch is installed.\n")
        sys.exit(1)

    # Load defaults from config if present
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    model_name = cfg.get("model_name", "Davlan/afro-xlmr-base")
    use_crf = cfg.get("use_crf", True)
    word_level = cfg.get("word_level", True)
    num_recurrent_steps = cfg.get("num_recurrent_steps", 2)
    relation_threshold = cfg.get("relation_threshold", 0.1)

    threshold = args.threshold
    if threshold is None:
        threshold = cfg.get("pair_accept_threshold", 0.5)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        out_dir = cfg.get("output_dir", "results/stage_aope_sdrn/afroxlmr_gemini_run1")
        ckpt_path = os.path.join(out_dir, "best_model.pt")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        checkpoint_path=ckpt_path,
        use_crf=use_crf,
        num_recurrent_steps=num_recurrent_steps,
        relation_threshold=relation_threshold,
        device=device,
    )

    if args.interactive:
        run_interactive(model, tokenizer, device, threshold, word_level)
    elif args.text:
        pred = predict_sentence(
            model=model,
            tokenizer=tokenizer,
            text=args.text,
            device=device,
            threshold=threshold,
            word_level=word_level,
        )
        print_prediction_card(pred)
    else:
        # Batch evaluation on example file or user-provided file
        test_file = args.file if args.file else "data/example_test_sentences.jsonl"
        print(f"\nRunning batch evaluation on: {test_file}")
        if not os.path.exists(test_file):
            print(f"Error: test file '{test_file}' does not exist.")
            return

        with open(test_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                expected = None
                if line.startswith("{"):
                    try:
                        record = json.loads(line)
                        sentence_text = record.get("text", "")
                        expected = record
                    except json.JSONDecodeError:
                        sentence_text = line
                else:
                    sentence_text = line

                pred = predict_sentence(
                    model=model,
                    tokenizer=tokenizer,
                    text=sentence_text,
                    device=device,
                    threshold=threshold,
                    word_level=word_level,
                )
                print_prediction_card(pred, expected=expected, index=idx)


if __name__ == "__main__":
    main()
