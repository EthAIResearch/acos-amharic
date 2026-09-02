"""
Stage 3 training: category classification (22-class, class-weighted).

Usage (in your GPU environment):
    pip install -r requirements.txt
    python train.py --config ../../configs/stage3_afroxlmr.yaml

Reports per-category P/R/F1 (not just macro/micro averages) -- with 10
low-support categories and 2 with zero explicit-pair training examples,
averaged metrics alone would hide exactly the failure modes that matter.
"""
import argparse
import json
import os
import random
import numpy as np
import torch
import yaml
from collections import Counter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import CategoryPairDataset, collate_fn
from model import PairClassifier

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pair_utils import compute_class_weights


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config_defaults(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    flat = {"model_name": cfg.get("model_name"), "output_dir": cfg.get("output_dir")}
    data = cfg.get("data", {})
    flat.update({"train": data.get("train"), "test": data.get("test"),
                 "label_space": data.get("label_space"), "max_length": data.get("max_length")})
    training = cfg.get("training", {})
    flat.update({"epochs": training.get("epochs"), "batch_size": training.get("batch_size"),
                 "lr": training.get("lr"), "warmup_ratio": training.get("warmup_ratio"),
                 "seed": training.get("seed"), "class_weight_cap": training.get("class_weight_cap")})
    return {k: v for k, v in flat.items() if v is not None}


def per_class_prf(y_true, y_pred, id2label):
    labels = sorted(id2label)
    counts_tp = Counter()
    counts_fp = Counter()
    counts_fn = Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            counts_tp[t] += 1
        else:
            counts_fp[p] += 1
            counts_fn[t] += 1

    report = {}
    macro_f1_sum = 0.0
    for lid in labels:
        tp, fp, fn = counts_tp[lid], counts_fp[lid], counts_fn[lid]
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[id2label[lid]] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        macro_f1_sum += f1
    macro_f1 = macro_f1_sum / len(labels) if labels else 0.0

    total_correct = sum(counts_tp.values())
    accuracy = total_correct / len(y_true) if y_true else 0.0
    return report, macro_f1, accuracy


@torch.no_grad()
def evaluate(model, dataset, device, id2label, batch_size=32):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**{k: v for k, v in batch.items() if k != "label"})
        preds = out["logits"].argmax(-1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["label"].cpu().tolist())
    report, macro_f1, accuracy = per_class_prf(y_true, y_pred, id2label)
    return {"per_category": report, "macro_f1": macro_f1, "accuracy": accuracy}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, remaining_argv = pre.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config) if pre_args.config else {}

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--train", required="train" not in config_defaults)
    ap.add_argument("--test", required="test" not in config_defaults)
    ap.add_argument("--label_space", required="label_space" not in config_defaults,
                     help="path to label_space.json (defines the fixed category list)")
    ap.add_argument("--model_name", default="Davlan/afro-xlmr-base")
    ap.add_argument("--output_dir", default="./stage3_category_ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight_cap", type=float, default=15.0)
    ap.set_defaults(**config_defaults)
    args = ap.parse_args(remaining_argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(args.label_space, encoding="utf-8") as f:
        label_space = json.load(f)
    categories = label_space["categories"]
    label2id = {c: i for i, c in enumerate(categories)}
    id2label = {i: c for c, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = CategoryPairDataset(args.train, tokenizer, label2id, max_length=args.max_length)
    test_ds = CategoryPairDataset(args.test, tokenizer, label2id, max_length=args.max_length)

    print(f"Train examples: {len(train_ds)} (skipped {train_ds.skipped_unknown_label} unknown-label, "
          f"{train_ds.skipped_truncated} truncated-span)")
    print(f"Test examples: {len(test_ds)} (skipped {test_ds.skipped_unknown_label} unknown-label, "
          f"{test_ds.skipped_truncated} truncated-span)")

    zero_support = [c for c in categories if not any(lbl == label2id[c] for _, _, _, lbl in train_ds.examples)]
    if zero_support:
        print(f"WARNING -- categories with ZERO training examples in this explicit-pairs subset "
              f"(structurally unlearnable, deferred to Stage 5): {zero_support}")

    train_label_counts = Counter(lbl for _, _, _, lbl in train_ds.examples)
    weight_by_label = compute_class_weights(
        {id2label[k]: v for k, v in train_label_counts.items()}, len(categories), cap=args.class_weight_cap
    )
    class_weights = [weight_by_label.get(id2label[i], args.class_weight_cap) for i in range(len(categories))]

    model = PairClassifier(args.model_name, num_labels=len(categories), class_weights=class_weights).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
    )

    best_macro_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item())

        metrics = evaluate(model, test_ds, device, id2label)
        print(f"\n[epoch {epoch+1}] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  -> new best (macro F1={best_macro_f1:.4f}), checkpoint saved")

    print(f"\nDone. Best macro F1 = {best_macro_f1:.4f}. Checkpoint: {args.output_dir}/best_model.pt")
    print("Per-category breakdown of best checkpoint saved to best_metrics.json -- "
          "check low-support categories specifically, not just the macro average.")


if __name__ == "__main__":
    main()
