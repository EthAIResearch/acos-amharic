"""
Stage 5 datasets: implicit aspect/opinion detection and classification.

Three sub-problems, quantified against real data in
docs/stage5_implicit_analysis.md:
  A. Opinion-anchored: given an explicit opinion span, does its aspect
     counterpart exist as a span, or is it implicit? (9,693 positive /
     23,617 negative examples in train)
  B. Aspect-anchored: given an explicit aspect span, is its opinion
     implicit? (4,560 positive / 23,617 negative)
  C. Fully-implicit: does this sentence contain a quad with NEITHER side
     explicit at all? (3,354 quads / 3,324 sentences, 8.5% of train)

For each sub-problem there are two dataset roles:
  - *DetectionDataset: binary classification (does the implicit
    counterpart exist?)
  - *CategorySentimentDataset: classify category or sentiment for the
    confirmed-implicit cases (label_field is "category" or "sentiment",
    label2id passed in so the same class serves both)

All reuse PairClassifier's masked-mean-pooling architecture (see
src/common/pair_model.py) -- an all-zero mask for the implicit side
correctly falls back to relying on the sentence-pooled representation,
so no architecture change was needed, only new dataset construction.
"""
import json
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pair_utils import (
    build_span_mask,
    is_implicit_side,
    quad_kind,
    word_span_to_subword_range,
)


def _is_implicit(q, side):
    return is_implicit_side(q, side)


def _quad_kind(q):
    return quad_kind(q)


class AnchoredDetectionDataset(Dataset):
    """Sub-problems A and B: binary detection of an implicit counterpart,
    anchored on the explicit side. anchor_side='opinion' -> detect implicit
    aspect (sub-problem A); anchor_side='aspect' -> detect implicit opinion
    (sub-problem B)."""

    def __init__(self, jsonl_path: str, tokenizer, anchor_side: str, max_length: int = 256):
        assert anchor_side in ("aspect", "opinion")
        self.tokenizer = tokenizer
        self.anchor_side = anchor_side
        self.other_side = "opinion" if anchor_side == "aspect" else "aspect"
        self.max_length = max_length
        self.examples = []  # (tokens, anchor_span, label)
        self.skipped_truncated = 0

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                tokens = rec["tokens"]
                for q in rec["quads"]:
                    if _is_implicit(q, anchor_side[0]):
                        continue  # anchor side itself must be explicit
                    anchor_span = (q[f"{anchor_side[0]}_start"], q[f"{anchor_side[0]}_end"])
                    label = 1 if _is_implicit(q, self.other_side[0]) else 0
                    self.examples.append((tokens, anchor_span, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, anchor_span, label = self.examples[idx]
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                              max_length=self.max_length, padding="max_length")
        word_ids = enc.word_ids(batch_index=0)
        anchor_range = word_span_to_subword_range(word_ids, *anchor_span)
        if anchor_range is None:
            self.skipped_truncated += 1

        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        anchor_mask = build_span_mask(self.max_length, anchor_range)
        zero_mask = [0] * self.max_length
        item["aspect_mask"] = torch.tensor(anchor_mask if self.anchor_side == "aspect" else zero_mask)
        item["opinion_mask"] = torch.tensor(anchor_mask if self.anchor_side == "opinion" else zero_mask)
        item["label"] = torch.tensor(label)
        return item


class AnchoredCategorySentimentDataset(Dataset):
    """Category/sentiment classification for confirmed-implicit quads,
    conditioned on the explicit anchor side. Only includes quads where
    exactly the `other_side` (relative to anchor) is implicit -- i.e. the
    positive examples from AnchoredDetectionDataset."""

    def __init__(self, jsonl_path: str, tokenizer, anchor_side: str, label_field: str,
                 label2id: dict, max_length: int = 256):
        assert anchor_side in ("aspect", "opinion")
        assert label_field in ("category", "sentiment")
        self.tokenizer = tokenizer
        self.anchor_side = anchor_side
        self.max_length = max_length
        self.label2id = label2id
        self.examples = []
        self.skipped_truncated = 0
        self.skipped_unknown_label = 0
        other_side = "opinion" if anchor_side == "aspect" else "aspect"

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                tokens = rec["tokens"]
                for q in rec["quads"]:
                    if _quad_kind(q) != f"implicit_{other_side}_only":
                        continue
                    if q[label_field] not in label2id:
                        self.skipped_unknown_label += 1
                        continue
                    anchor_span = (q[f"{anchor_side[0]}_start"], q[f"{anchor_side[0]}_end"])
                    self.examples.append((tokens, anchor_span, label2id[q[label_field]]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, anchor_span, label = self.examples[idx]
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                              max_length=self.max_length, padding="max_length")
        word_ids = enc.word_ids(batch_index=0)
        anchor_range = word_span_to_subword_range(word_ids, *anchor_span)
        if anchor_range is None:
            self.skipped_truncated += 1

        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        anchor_mask = build_span_mask(self.max_length, anchor_range)
        zero_mask = [0] * self.max_length
        item["aspect_mask"] = torch.tensor(anchor_mask if self.anchor_side == "aspect" else zero_mask)
        item["opinion_mask"] = torch.tensor(anchor_mask if self.anchor_side == "opinion" else zero_mask)
        item["label"] = torch.tensor(label)
        return item


class FullyImplicitDetectionDataset(Dataset):
    """Sub-problem C: sentence-level binary detection -- does this sentence
    contain at least one quad with neither side explicit? No span anchor
    exists at all; both masks are always zero (classification falls back
    entirely to the sentence-pooled representation)."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []  # (tokens, label)
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                has_fully_implicit = any(_quad_kind(q) == "fully_implicit" for q in rec["quads"])
                self.examples.append((rec["tokens"], int(has_fully_implicit)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, label = self.examples[idx]
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                              max_length=self.max_length, padding="max_length")
        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        zero_mask = [0] * self.max_length
        item["aspect_mask"] = torch.tensor(zero_mask)
        item["opinion_mask"] = torch.tensor(zero_mask)
        item["label"] = torch.tensor(label)
        return item


class FullyImplicitCategorySentimentDataset(Dataset):
    """Category/sentiment classification for fully-implicit quads. Only
    ~30 of 3,324 train sentences have 2+ fully-implicit quads (see
    docs/stage5_implicit_analysis.md); this dataset takes the first such
    quad per sentence as a documented simplification."""

    def __init__(self, jsonl_path: str, tokenizer, label_field: str, label2id: dict, max_length: int = 256):
        assert label_field in ("category", "sentiment")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.skipped_unknown_label = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for q in rec["quads"]:
                    if _quad_kind(q) != "fully_implicit":
                        continue
                    if q[label_field] not in label2id:
                        self.skipped_unknown_label += 1
                        continue
                    self.examples.append((rec["tokens"], label2id[q[label_field]]))
                    break  # first fully-implicit quad only

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, label = self.examples[idx]
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                              max_length=self.max_length, padding="max_length")
        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        zero_mask = [0] * self.max_length
        item["aspect_mask"] = torch.tensor(zero_mask)
        item["opinion_mask"] = torch.tensor(zero_mask)
        item["label"] = torch.tensor(label)
        return item


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
