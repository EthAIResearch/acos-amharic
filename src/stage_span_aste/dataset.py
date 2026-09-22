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
    build_gold_span_labels,
    enumerate_spans,
    extract_explicit_triplets,
    RELATION2ID,
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
        n_words = len(tokens)
        quads = rec.get("quads", [])

        # Enumerate spans up to max_span_length words
        spans = enumerate_spans(n_words, max_span_length=self.max_span_length)

        # Build gold mention labels: 0: INVALID, 1: TARGET, 2: OPINION
        mention_labels = build_gold_span_labels(spans, quads)

        # Build gold pairs: list of ((a_s, a_e), (o_s, o_e), rel_id)
        gold_triplets = extract_explicit_triplets(quads)
        gold_pairs = [
            (a_span, o_span, RELATION2ID.get(senti, RELATION2ID["INVALID"]))
            for a_span, o_span, senti in gold_triplets
        ]

        # Tokenize with subword-to-word alignment
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        word_ids = enc.word_ids(batch_index=0)

        # Build vectorized subword-to-word mean pooling matrix P (M x L)
        L = self.max_length
        M = self.max_words
        word_to_subwords = [[] for _ in range(M)]
        for si, wi in enumerate(word_ids):
            if wi is not None and wi < M:
                word_to_subwords[wi].append(si)

        subword_to_word = [[0.0] * L for _ in range(M)]
        word_mask = [False] * M

        for wi in range(min(n_words, M)):
            sis = word_to_subwords[wi]
            if sis:
                word_mask[wi] = True
                inv_len = 1.0 / len(sis)
                for si in sis:
                    subword_to_word[wi][si] = inv_len

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "subword_to_word": torch.tensor(subword_to_word, dtype=torch.float32),
            "word_mask": torch.tensor(word_mask, dtype=torch.bool),
            "num_words": n_words,
            "spans": spans,
            "mention_labels": torch.tensor(mention_labels, dtype=torch.long),
            "gold_pairs": gold_pairs,
            "tokens": tokens,
            "raw_quads": quads,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function dynamically slicing word pooling matrices to the
    maximum active words in the current batch.
    """
    # Find max active words across batch
    max_m = 1
    for b in batch:
        nz = b["word_mask"].nonzero()
        if len(nz) > 0:
            max_m = max(max_m, int(nz[-1].item()) + 1)

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    subword_to_word = torch.stack([b["subword_to_word"][:max_m] for b in batch])

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
