"""
Span-ASTE PyTorch Dataset and Dynamic Collate Function
======================================================
Prepares input tokens, subword-to-word pooling matrices, enumerated spans,
and gold mention/pair targets for Span-ASTE.
"""
import json
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from span_utils import (
    RELATION2ID,
    build_gold_span_labels,
    enumerate_spans,
    extract_explicit_triplets,
)


class SpanASTEDataset(Dataset):
    """
    Dataset for Span-ASTE. Enumerates candidate spans and constructs
    gold mention labels and gold triplet relations.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = 256,
        max_words: int = 128,
        max_span_length: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_words = max_words
        self.max_span_length = max_span_length
        self.records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        tokens = rec["tokens"]
        quads = rec.get("quads", [])

        # Tokenize with subword-to-word alignment
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        word_ids = enc.word_ids(batch_index=0)

        # Determine valid words that were actually encoded into subwords
        valid_words = [w for w in word_ids if w is not None]
        num_encoded_words = (max(valid_words) + 1) if valid_words else 0
        num_encoded_words = min(num_encoded_words, self.max_words, len(tokens))

        # Enumerate candidate spans strictly within the encoded words
        spans = enumerate_spans(num_encoded_words, max_span_length=self.max_span_length)

        # Build gold mention labels: 0: INVALID, 1: TARGET, 2: OPINION
        mention_labels = build_gold_span_labels(spans, quads)

        # Build gold pairs: only keep pairs within the encoded word boundary
        gold_triplets = extract_explicit_triplets(quads)
        gold_pairs = [
            (a_span, o_span, RELATION2ID.get(senti, RELATION2ID["INVALID"]))
            for a_span, o_span, senti in gold_triplets
            if a_span[1] <= num_encoded_words and o_span[1] <= num_encoded_words
        ]

        # Build vectorized subword-to-word mean pooling matrix P (num_encoded_words x L)
        L = self.max_length
        M = num_encoded_words
        word_to_subwords = [[] for _ in range(M)]
        for si, wi in enumerate(word_ids):
            if wi is not None and wi < M:
                word_to_subwords[wi].append(si)

        subword_to_word = [[0.0] * L for _ in range(M)]
        for wi in range(M):
            sis = word_to_subwords[wi]
            if sis:
                inv_len = 1.0 / len(sis)
                for si in sis:
                    subword_to_word[wi][si] = inv_len

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "subword_to_word": torch.tensor(subword_to_word, dtype=torch.float32),  # (M, L)
            "num_words": num_encoded_words,
            "spans": spans,
            "mention_labels": torch.tensor(mention_labels, dtype=torch.long),
            "gold_pairs": gold_pairs,
            "tokens": tokens[:num_encoded_words],
            "raw_quads": quads,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function dynamically padding word pooling matrices to the
    maximum active words in the current batch.
    """
    max_batch_words = max([b["num_words"] for b in batch], default=0)
    max_batch_words = max(max_batch_words, 1)  # Ensure at least 1 row to prevent empty tensors
    L = batch[0]["input_ids"].size(0)

    # Pad each sample's subword_to_word to (max_batch_words, L)
    padded_s2w = []
    for b in batch:
        s2w = b["subword_to_word"]
        m = s2w.size(0)
        if m < max_batch_words:
            padding = torch.zeros((max_batch_words - m, L), dtype=torch.float32)
            padded = torch.cat([s2w, padding], dim=0)
        else:
            padded = s2w[:max_batch_words]
        padded_s2w.append(padded)

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    subword_to_word = torch.stack(padded_s2w)  # (B, max_batch_words, L)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "subword_to_word": subword_to_word,
        "num_words": [b["num_words"] for b in batch],
        "seq_spans": [b["spans"] for b in batch],
        "gold_mention_labels": [b["mention_labels"] for b in batch],
        "gold_pairs": [b["gold_pairs"] for b in batch],
        "tokens": [b["tokens"] for b in batch],
        "raw_quads": [b["raw_quads"] for b in batch],
    }
