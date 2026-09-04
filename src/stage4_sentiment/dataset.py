"""
Stage 4 dataset: sentiment classification (NEUTRAL/POSITIVE/NEGATIVE) over
(sentence, aspect_span, opinion_span) triples. Trains only on explicit-both
quads, same scope as Stage 3.

Unlike Stage 3, the label space here is fixed and small (3 classes, always
present) -- no zero-support-category bookkeeping needed. The real challenge
is NEUTRAL: only 3.1% of explicit-pair examples (741 of 23,617 in train),
noticeably rarer than its 8.7% share of the full dataset -- implicit-opinion
cases disproportionately carry neutral sentiment. See
docs/stage4_sentiment_analysis.md.
"""
import json
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pair_utils import (
    build_span_mask,
    explicit_quads,
    word_span_to_subword_range,
)

SENTIMENT_LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]  # fixed, alphabetical for determinism
LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


class SentimentPairDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.skipped_truncated = 0
        self.skipped_unknown_label = 0

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                tokens = rec["tokens"]
                for q in explicit_quads(rec["quads"]):
                    sentiment = q["sentiment"]
                    if sentiment not in LABEL2ID:
                        self.skipped_unknown_label += 1
                        continue
                    self.examples.append((
                        tokens,
                        (q["a_start"], q["a_end"]),
                        (q["o_start"], q["o_end"]),
                        LABEL2ID[sentiment],
                    ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, a_span, o_span, label_id = self.examples[idx]

        enc = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True,
            max_length=self.max_length, padding="max_length",
        )
        word_ids = enc.word_ids(batch_index=0)

        a_range = word_span_to_subword_range(word_ids, *a_span)
        o_range = word_span_to_subword_range(word_ids, *o_span)
        if a_range is None or o_range is None:
            self.skipped_truncated += 1

        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        item["aspect_mask"] = torch.tensor(build_span_mask(self.max_length, a_range))
        item["opinion_mask"] = torch.tensor(build_span_mask(self.max_length, o_range))
        item["label"] = torch.tensor(label_id)
        return item


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
