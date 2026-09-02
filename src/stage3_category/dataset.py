"""
Stage 3 dataset: category classification over (sentence, aspect_span,
opinion_span) triples. Trains only on explicit-both quads -- see
docs/stage3_category_analysis.md for why 2 of the 22 categories
(PUBLIC_SERVICES#COMMUNITY_SUPPORT, PUBLIC_SERVICES#INFRASTRUCTURE) have
zero examples here and are structurally deferred to Stage 5.
"""
import json
import torch
from torch.utils.data import Dataset

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pair_utils import build_span_mask, explicit_quads, word_span_to_subword_range


class CategoryPairDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, label2id: dict, max_length: int = 256):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.examples = []  # (tokens, a_span, o_span, label_id)
        self.skipped_truncated = 0
        self.skipped_unknown_label = 0

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                tokens = rec["tokens"]
                for q in explicit_quads(rec["quads"]):
                    if q["category"] not in label2id:
                        self.skipped_unknown_label += 1
                        continue
                    self.examples.append((
                        tokens,
                        (q["a_start"], q["a_end"]),
                        (q["o_start"], q["o_end"]),
                        label2id[q["category"]],
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
            # fall back to an all-zero mask -- rare (long-sentence truncation);
            # the example still trains on sentence context, just without a
            # located span. Flagged via self.skipped_truncated for reporting.

        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        item["aspect_mask"] = torch.tensor(build_span_mask(self.max_length, a_range))
        item["opinion_mask"] = torch.tensor(build_span_mask(self.max_length, o_range))
        item["label"] = torch.tensor(label_id)
        return item


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
