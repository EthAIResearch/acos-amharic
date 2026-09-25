"""
PyTorch Dataset and Dynamic Collation for ByT5 ACOS Quadruple Extraction
========================================================================
Loads Amharic ACOS jsonl data, applies byte-level tokenization for ByT5,
and prepares source-target pairs with label padding (-100).
"""
import json
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from linearization import extract_gold_quad_tuples, quads_to_target


class ByT5ACOSDataset(Dataset):
    """
    PyTorch Dataset for ByT5 ACOS Seq2Seq training and evaluation.
    Encodes raw Amharic text into byte IDs and formats gold quadruples into
    linearized target byte sequences.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_source_length: int = 768,
        max_target_length: int = 384,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset file not found at: {jsonl_path}")

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data.get("text", "").strip()
                tokens = data.get("tokens", [])
                quads = data.get("quads", [])

                target_str = quads_to_target(quads, tokens)
                gold_tuples = extract_gold_quad_tuples(data)

                # Tokenize source sentence (ByT5 encodes directly into UTF-8 byte IDs)
                source_enc = tokenizer(
                    text,
                    max_length=self.max_source_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                # Tokenize target sequence
                target_enc = tokenizer(
                    target_str,
                    max_length=self.max_target_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                self.examples.append({
                    "input_ids": source_enc["input_ids"],
                    "attention_mask": source_enc["attention_mask"],
                    "labels": target_enc["input_ids"],
                    "raw_text": text,
                    "raw_target": target_str,
                    "gold_quads": gold_tuples,
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """
    Dynamically pads input_ids and labels to the maximum length in the batch.
    Pads labels with -100 so that PyTorch CrossEntropyLoss ignores them.
    """
    max_src_len = max(len(x["input_ids"]) for x in batch)
    max_tgt_len = max(len(x["labels"]) for x in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    raw_texts = []
    raw_targets = []
    gold_quads = []

    for item in batch:
        src_ids = item["input_ids"]
        att_mask = item["attention_mask"]
        tgt_ids = item["labels"]

        src_pad_len = max_src_len - len(src_ids)
        tgt_pad_len = max_tgt_len - len(tgt_ids)

        padded_input_ids = src_ids + [pad_token_id] * src_pad_len
        padded_attention_mask = att_mask + [0] * src_pad_len
        padded_labels = tgt_ids + [-100] * tgt_pad_len

        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)
        batch_labels.append(padded_labels)
        raw_texts.append(item["raw_text"])
        raw_targets.append(item["raw_target"])
        gold_quads.append(item["gold_quads"])

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "raw_texts": raw_texts,
        "raw_targets": raw_targets,
        "gold_quads": gold_quads,
    }
