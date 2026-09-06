"""
Stage 5 training: implicit aspect/opinion detection and classification.

One parameterized script covers all sub-tasks, since they share the exact
same architecture (PairClassifier with the zero-mask fallback) and training
loop (AMP, logit adjustment, per-class F1) already built for Stage 3/4:

  --subtask detect_implicit_aspect   binary, anchor=opinion  (9,693 pos / 23,617 neg)
  --subtask detect_implicit_opinion  binary, anchor=aspect   (4,560 pos / 23,617 neg)
  --subtask detect_fully_implicit    binary, sentence-level  (3,324 pos / 35,937 neg sentences)
  --subtask category  --anchor {opinion,aspect,none}   22-way, on the confirmed-implicit subset
  --subtask sentiment --anchor {opinion,aspect,none}   3-way,  on the confirmed-implicit subset

Usage:
    python train.py --config ../../configs/stage5_detect_implicit_aspect.yaml
    python train.py --config ../../configs/stage5_category_opinion_anchor.yaml
    # etc. -- one config per sub-task, same pattern as Stage 3/4.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from dataset import (
    AnchoredCategorySentimentDataset,
    AnchoredDetectionDataset,
    FullyImplicitCategorySentimentDataset,
    FullyImplicitDetectionDataset,
    collate_fn,
)
from model import PairClassifier
from pair_utils import compute_class_weights, compute_log_priors

BINARY_LABELS = {0: "no_implicit_counterpart", 1: "has_implicit_counterpart"}
SENTIMENT_LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config_defaults(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    flat = {"model_name": cfg.get("model_name"), "output_dir": cfg.get("output_dir"),
            "subtask": cfg.get("subtask"), "anchor": cfg.get("anchor")}
    data = cfg.get("data", {})
    flat.update({"train": data.get("train"), "test": data.get("test"),
                 "label_space": data.get("label_space"), "max_length": data.get("max_length")})
    training = cfg.get("training", {})
    flat.update({"epochs": training.get("epochs"), "batch_size": training.get("batch_size"),
                 "lr": training.get("lr"), "warmup_ratio": training.get("warmup_ratio"),
                 "seed": training.get("seed"), "class_weight_cap": training.get("class_weight_cap"),
                 "weight_decay": training.get("weight_decay"), "fp16": training.get("fp16")})
    flat["loss_type"] = cfg.get("loss_type")
    flat["logit_adjustment_tau"] = cfg.get("logit_adjustment_tau")
    return {k: v for k, v in flat.items() if v is not None}


def per_class_prf(y_true, y_pred, id2label):
    """Same logic as Stage 3/4 -- see tests/test_stage3_metrics.py."""
    labels = sorted(id2label)
    counts_tp, counts_fp, counts_fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            counts_tp[t] += 1
        else:
            counts_fp[p] += 1
            counts_fn[t] += 1
    report = {}
    present_f1s = []
    for lid in labels:
        tp, fp, fn = counts_tp[lid], counts_fp[lid], counts_fn[lid]
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[id2label[lid]] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support > 0:
            present_f1s.append(f1)
    all_f1s = [report[id2label[lid]]["f1"] for lid in labels]
    macro_f1_all = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    macro_f1_present = sum(present_f1s) / len(present_f1s) if present_f1s else 0.0
    accuracy = sum(counts_tp.values()) / len(y_true) if y_true else 0.0
    n_absent = len(labels) - len(present_f1s)
    return report, macro_f1_all, macro_f1_present, n_absent, accuracy


def build_datasets(args, tokenizer, label2id=None):
    """Returns (train_ds, test_ds, id2label) for the requested subtask."""
    if args.subtask == "detect_implicit_aspect":
        anchor = "opinion"
        train_ds = AnchoredDetectionDataset(args.train, tokenizer, anchor, args.max_length)
        test_ds = AnchoredDetectionDataset(args.test, tokenizer, anchor, args.max_length)
        return train_ds, test_ds, BINARY_LABELS

    if args.subtask == "detect_implicit_opinion":
        anchor = "aspect"
        train_ds = AnchoredDetectionDataset(args.train, tokenizer, anchor, args.max_length)
        test_ds = AnchoredDetectionDataset(args.test, tokenizer, anchor, args.max_length)
        return train_ds, test_ds, BINARY_LABELS

    if args.subtask == "detect_fully_implicit":
        train_ds = FullyImplicitDetectionDataset(args.train, tokenizer, args.max_length)
        test_ds = FullyImplicitDetectionDataset(args.test, tokenizer, args.max_length)
        return train_ds, test_ds, BINARY_LABELS

    if args.subtask in ("category", "sentiment"):
        if args.subtask == "sentiment":
            id2label = {i: lbl for i, lbl in enumerate(SENTIMENT_LABELS)}
            label2id_local = {lbl: i for i, lbl in id2label.items()}
        else:
            id2label = {i: c for c, i in label2id.items()}
            label2id_local = label2id

        if args.anchor == "none":
            train_ds = FullyImplicitCategorySentimentDataset(args.train, tokenizer, args.subtask, label2id_local, args.max_length)
            test_ds = FullyImplicitCategorySentimentDataset(args.test, tokenizer, args.subtask, label2id_local, args.max_length)
        else:
            train_ds = AnchoredCategorySentimentDataset(args.train, tokenizer, args.anchor, args.subtask, label2id_local, args.max_length)
            test_ds = AnchoredCategorySentimentDataset(args.test, tokenizer, args.anchor, args.subtask, label2id_local, args.max_length)
        return train_ds, test_ds, id2label

    raise ValueError(f"Unknown subtask: {args.subtask}")


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
    report, macro_f1_all, macro_f1_present, n_absent, accuracy = per_class_prf(y_true, y_pred, id2label)
    return {
        "per_label": report,
        "macro_f1_all_categories": macro_f1_all,
        "macro_f1_present_categories": macro_f1_present,
        "n_categories_absent_from_eval_set": n_absent,
        "accuracy": accuracy,
    }


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, remaining_argv = pre.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config) if pre_args.config else {}

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--subtask", required="subtask" not in config_defaults,
                     choices=["detect_implicit_aspect", "detect_implicit_opinion",
                              "detect_fully_implicit", "category", "sentiment"])
    ap.add_argument("--anchor", choices=["aspect", "opinion", "none"], default="none",
                     help="Only used when --subtask is category/sentiment.")
    ap.add_argument("--train", required="train" not in config_defaults)
    ap.add_argument("--test", required="test" not in config_defaults)
    ap.add_argument("--label_space", default=None,
                     help="Required when --subtask category (defines the 22-category label space).")
    ap.add_argument("--model_name", default="Davlan/afro-xlmr-base")
    ap.add_argument("--output_dir", default="./stage5_ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight_cap", type=float, default=15.0)
    ap.add_argument("--loss_type", choices=["class_weighted", "logit_adjustment"], default="logit_adjustment")
    ap.add_argument("--logit_adjustment_tau", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.set_defaults(**config_defaults)
    args = ap.parse_args(remaining_argv)

    if args.subtask == "category" and not args.label_space:
        raise SystemExit("--label_space is required for --subtask category")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    label2id = None
    if args.subtask == "category":
        with open(args.label_space, encoding="utf-8") as f:
            label2id = {c: i for i, c in enumerate(json.load(f)["categories"])}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds, test_ds, id2label = build_datasets(args, tokenizer, label2id)
    num_labels = len(id2label)

    print(f"Subtask: {args.subtask}" + (f" (anchor={args.anchor})" if args.subtask in ("category", "sentiment") else ""))
    print(f"Train examples: {len(train_ds)} | Test examples: {len(test_ds)} | num_labels: {num_labels}")

    train_label_counts = Counter(ex[-1] for ex in train_ds.examples)
    print("Train label distribution:", {id2label[k]: v for k, v in train_label_counts.items()})

    if args.loss_type == "logit_adjustment":
        log_priors = compute_log_priors(dict(train_label_counts), num_labels)
        model = PairClassifier(args.model_name, num_labels=num_labels,
                                log_priors=log_priors, tau=args.logit_adjustment_tau).to(device)
        print(f"Using logit-adjusted loss (tau={args.logit_adjustment_tau})")
    else:
        weight_by_label = compute_class_weights(
            {id2label[k]: v for k, v in train_label_counts.items()}, num_labels, cap=args.class_weight_cap
        )
        class_weights = [weight_by_label.get(id2label[i], args.class_weight_cap) for i in range(num_labels)]
        model = PairClassifier(args.model_name, num_labels=num_labels, class_weights=class_weights).to(device)
        print(f"Using class-weighted loss (cap={args.class_weight_cap})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
    )

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if args.fp16 and not use_amp:
        print("--fp16 requested but no CUDA device available -- running in full precision on CPU.")

    best_macro_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(**batch)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix(loss=loss.item())

        metrics = evaluate(model, test_ds, device, id2label)
        print(f"\n[epoch {epoch+1}] accuracy={metrics['accuracy']:.4f} "
              f"macro_f1={metrics['macro_f1_present_categories']:.4f}")

        if metrics["macro_f1_present_categories"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1_present_categories"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  -> new best (macro F1={best_macro_f1:.4f}), checkpoint saved")

    print(f"\nDone. Best macro F1 = {best_macro_f1:.4f}. Checkpoint: {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
