"""
Shared helpers for loading trained checkpoints and running span-conditioned
classification at inference time. Used by run_inference.py to orchestrate
Stages 1, 3, 4, and 5 (Stage 2 is the pure heuristic from
stage2_pairing/candidates.py -- no checkpoint to load).

Each stage's train.py already writes resolved_args.json (containing
model_name) into its output_dir alongside best_model.pt and the saved
tokenizer -- load_pair_classifier reads that so the caller never has to
redundantly specify which backbone a checkpoint used.
"""
import json
import os
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pair_model import PairClassifier
from pair_utils import word_span_to_subword_range, build_span_mask


def load_pair_classifier(ckpt_dir: str, num_labels: int, device):
    """Loads a PairClassifier checkpoint (used by Stage 3, 4, and every
    Stage 5 sub-model -- they all share this architecture)."""
    from transformers import AutoTokenizer

    with open(os.path.join(ckpt_dir, "resolved_args.json"), encoding="utf-8") as f:
        model_name = json.load(f)["model_name"]

    model = PairClassifier(model_name, num_labels=num_labels)
    state_dict = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    return model, tokenizer


def load_stage1_model(ckpt_dir: str, device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_tagging"))
    from model import JointTaggingModel
    from transformers import AutoTokenizer

    with open(os.path.join(ckpt_dir, "resolved_args.json"), encoding="utf-8") as f:
        model_name = json.load(f)["model_name"]

    model = JointTaggingModel(model_name)
    state_dict = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    return model, tokenizer


@torch.no_grad()
def extract_spans(model, tokenizer, tokens: list, device, max_length: int = 256):
    """Stage 1 inference: tokens -> (aspect_spans, opinion_spans), both
    lists of (start, end) word-index tuples."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
    from bio_labels import decode_bio_spans
    from align import decode_subword_predictions

    enc = tokenizer(tokens, is_split_into_words=True, truncation=True,
                     max_length=max_length, padding="max_length", return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    enc = {k: v.to(device) for k, v in enc.items() if k != "overflow_to_sample_mapping"}

    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    a_pred_ids = out["aspect_logits"].argmax(-1)[0].cpu().tolist()
    o_pred_ids = out["opinion_logits"].argmax(-1)[0].cpu().tolist()

    a_word_tags = decode_subword_predictions(word_ids, a_pred_ids[:len(word_ids)])
    o_word_tags = decode_subword_predictions(word_ids, o_pred_ids[:len(word_ids)])
    return decode_bio_spans(a_word_tags), decode_bio_spans(o_word_tags)


@torch.no_grad()
def classify_span_pair(model, tokenizer, tokens: list, device, id2label: dict,
                        aspect_span=None, opinion_span=None, max_length: int = 256):
    """Runs a PairClassifier on one (sentence, optional aspect_span,
    optional opinion_span) input. Pass None for either span to use an
    all-zero mask (the implicit-side / no-anchor case Stage 5 relies on).
    Returns the predicted label (already mapped via id2label)."""
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True,
                     max_length=max_length, padding="max_length", return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)

    zero_mask = [0] * max_length
    a_range = word_span_to_subword_range(word_ids, *aspect_span) if aspect_span else None
    o_range = word_span_to_subword_range(word_ids, *opinion_span) if opinion_span else None
    aspect_mask = build_span_mask(max_length, a_range) if aspect_span else zero_mask
    opinion_mask = build_span_mask(max_length, o_range) if opinion_span else zero_mask

    inputs = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "aspect_mask": torch.tensor([aspect_mask]).to(device),
        "opinion_mask": torch.tensor([opinion_mask]).to(device),
    }
    out = model(**inputs)
    pred_id = out["logits"].argmax(-1).item()
    return id2label[pred_id]
