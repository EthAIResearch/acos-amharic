"""
Stage 1 dataset: joint Aspect Term Extraction (ATE) + Opinion Term Extraction (OTE)
via BIO tagging, sharing one encoder with two independent linear heads.
Run where transformers/torch are installed.
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from align import IGNORE_INDEX, align_labels_to_subwords
from bio_labels import build_word_bio


class TaggingDataset(Dataset):
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

        a_spans = [(q["a_start"], q["a_end"]) for q in rec["quads"]]
        o_spans = [(q["o_start"], q["o_end"]) for q in rec["quads"]]
        a_word_tags = build_word_bio(n, a_spans)
        o_word_tags = build_word_bio(n, o_spans)

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        word_ids = enc.word_ids(batch_index=0)

        a_label_ids = align_labels_to_subwords(word_ids, a_word_tags)
        o_label_ids = align_labels_to_subwords(word_ids, o_word_tags)

        item = {k: torch.tensor(v) for k, v in enc.items() if k != "overflow_to_sample_mapping"}
        item["aspect_labels"] = torch.tensor(a_label_ids)
        item["opinion_labels"] = torch.tensor(o_label_ids)
        return item


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
