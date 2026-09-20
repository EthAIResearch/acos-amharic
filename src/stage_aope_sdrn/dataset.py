"""
Dataset for the joint SDRN-style AOPE model. Builds, per example:
  - aspect/opinion BIO labels aligned to subwords (reusing bio_labels.py /
    align.py, identical to Stage 1's tagging dataset)
  - a subword-level gold relation matrix (relation_utils.py), built from
    this project's explicit-both aspect-opinion pairs
"""
import json
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from align import align_labels_to_subwords
from bio_labels import build_word_bio
from relation_utils import build_word_relation_matrix, explicit_pairs


class JointAOPEDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tokens = rec["tokens"]
        n = len(tokens)
        pairs = explicit_pairs(rec["quads"])

        a_spans = [p[0] for p in pairs]
        o_spans = [p[1] for p in pairs]
        a_word_tags = build_word_bio(n, a_spans)
        o_word_tags = build_word_bio(n, o_spans)
        word_rel_matrix = build_word_relation_matrix(n, pairs)

        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                              max_length=self.max_length, padding="max_length")
        word_ids = enc.word_ids(batch_index=0)

        a_label_ids = align_labels_to_subwords(word_ids, a_word_tags)
        o_label_ids = align_labels_to_subwords(word_ids, o_word_tags)

        # Project the word-level relation matrix onto subwords: every
        # subword of word i inherits word i's relation row/column.
        L = self.max_length
        rel_matrix = [[0] * L for _ in range(L)]
        for si, wi in enumerate(word_ids):
            if wi is None:
                continue
            for sj, wj in enumerate(word_ids):
                if wj is None:
                    continue
                if wi < n and wj < n:
                    rel_matrix[si][sj] = word_rel_matrix[wi][wj]

        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        item["aspect_labels"] = torch.tensor(a_label_ids)
        item["opinion_labels"] = torch.tensor(o_label_ids)
        item["relation_labels"] = torch.tensor(rel_matrix)
        return item


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
