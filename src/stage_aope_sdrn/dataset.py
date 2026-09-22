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
from align import LABEL2ID_5WAY, align_labels_to_subwords, align_labels_to_subwords_5way
from bio_labels import build_word_bio, build_word_bio_5way
from relation_utils import build_word_relation_matrix, explicit_pairs


class JointAOPEDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256, max_words: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_words = max_words
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
        unified_word_tags = build_word_bio_5way(n, a_spans, o_spans)
        word_rel_matrix = build_word_relation_matrix(n, pairs)

        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                              max_length=self.max_length, padding="max_length")
        word_ids = enc.word_ids(batch_index=0)

        a_label_ids = align_labels_to_subwords(word_ids, a_word_tags)
        o_label_ids = align_labels_to_subwords(word_ids, o_word_tags)
        unified_label_ids = align_labels_to_subwords_5way(word_ids, unified_word_tags)

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

        # Phase 3 Word-Level SDRN structures
        M = self.max_words
        word_to_subwords = [[] for _ in range(M)]
        for si, wi in enumerate(word_ids):
            if wi is not None and wi < M:
                word_to_subwords[wi].append(si)

        subword_to_word = [[0.0] * L for _ in range(M)]
        word_mask = [False] * M
        word_labels = [-100] * M

        for wi in range(min(n, M)):
            sis = word_to_subwords[wi]
            if sis:
                word_mask[wi] = True
                word_labels[wi] = LABEL2ID_5WAY.get(unified_word_tags[wi], 0)
                inv_len = 1.0 / len(sis)
                for si in sis:
                    subword_to_word[wi][si] = inv_len

        word_rel_labels = [[-100] * M for _ in range(M)]
        for wi in range(min(n, M)):
            if not word_mask[wi]:
                continue
            for wj in range(min(n, M)):
                if word_mask[wj]:
                    word_rel_labels[wi][wj] = word_rel_matrix[wi][wj]

        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        item["content_mask"] = torch.tensor([wid is not None for wid in word_ids], dtype=torch.bool)
        item["labels"] = torch.tensor(unified_label_ids)
        item["aspect_labels"] = torch.tensor(a_label_ids)
        item["opinion_labels"] = torch.tensor(o_label_ids)
        item["relation_labels"] = torch.tensor(rel_matrix)

        # Word-level tensors (Phase 3)
        item["subword_to_word"] = torch.tensor(subword_to_word, dtype=torch.float32)
        item["word_mask"] = torch.tensor(word_mask, dtype=torch.bool)
        item["word_labels"] = torch.tensor(word_labels, dtype=torch.long)
        item["word_relation_labels"] = torch.tensor(word_rel_labels, dtype=torch.long)
        return item


def collate_fn(batch):
    # Dynamically slice word-level tensors to max words in this batch
    if "word_mask" in batch[0]:
        max_m = 1
        for b in batch:
            nz = b["word_mask"].nonzero()
            if len(nz) > 0:
                max_m = max(max_m, int(nz[-1].item()) + 1)
        collated = {}
        for k in batch[0]:
            if k in ("subword_to_word", "word_mask", "word_labels"):
                collated[k] = torch.stack([b[k][:max_m] for b in batch])
            elif k == "word_relation_labels":
                collated[k] = torch.stack([b[k][:max_m, :max_m] for b in batch])
            else:
                collated[k] = torch.stack([b[k] for b in batch])
        return collated
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
