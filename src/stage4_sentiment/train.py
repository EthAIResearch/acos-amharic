"""
Stage 4 training: sentiment classification (NEGATIVE/NEUTRAL/POSITIVE),
class-weighted or logit-adjusted loss.

Usage (in your GPU environment):
    pip install -r requirements.txt
    python train.py --config ../../configs/stage4_afroxlmr.yaml

NEUTRAL is the class to watch -- only 3.1% of training examples (741 of
23,617). ABSA literature consistently finds neutral is the hardest class
under imbalance (see docs/stage4_sentiment_analysis.md) -- if NEUTRAL
recall is near zero after training, that's the expected failure mode to
investigate first, not a sign something else is broken.
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

from dataset import SentimentPairDataset, collate_fn, SENTIMENT_LABELS, ID2LABEL
from model import PairClassifier

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pair_utils import compute_class_weights, compute_log_priors


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
    flat.update({"train": data.get("train"), "test": data.get("test"), "max_length": data.get("max_length")})
    training = cfg.get("training", {})
    flat.update({"epochs": training.get("epochs"), "batch_size": training.get("batch_size"),
                 "lr": training.get("lr"), "warmup_ratio": training.get("warmup_ratio"),
                 "seed": training.get("seed"), "class_weight_cap": training.get("class_weight_cap"),
                 "weight_decay": training.get("weight_decay"), "fp16": training.get("fp16")})
    flat["loss_type"] = cfg.get("loss_type")
    flat["logit_adjustment_tau"] = cfg.get("logit_adjustment_tau")
    return {k: v for k, v in flat.items() if v is not None}


def per_class_prf(y_true, y_pred, id2label):
    """Same logic as Stage 3's -- see tests/test_stage3_metrics.py for why
    this specific shape (returning both macro_f1_all and macro_f1_present)
    matters. Sentiment has all 3 classes present in every split here, so
    the two numbers should be identical in practice -- kept for consistency
    with Stage 3's reporting format and as a safety net if that ever changes."""
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
        "per_sentiment": report,
        "macro_f1": macro_f1_present,
        "macro_f1_all_sentiments": macro_f1_all,
        "macro_f1_present_sentiments": macro_f1_present,
        "macro_f1_all_categories": macro_f1_all,
        "macro_f1_present_categories": macro_f1_present,
        "n_sentiments_absent_from_eval_set": n_absent,
        "n_categories_absent_from_eval_set": n_absent,
        "accuracy": accuracy,
    }


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, remaining_argv = pre.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config) if pre_args.config else {}

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--train", required="train" not in config_defaults)
    ap.add_argument("--test", required="test" not in config_defaults)
    ap.add_argument("--model_name", default="Davlan/afro-xlmr-base")
    ap.add_argument("--output_dir", default="./stage4_sentiment_ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight_cap", type=float, default=15.0)
    ap.add_argument("--loss_type", choices=["class_weighted", "logit_adjustment"],
                     default="logit_adjustment")
    ap.add_argument("--logit_adjustment_tau", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.set_defaults(**config_defaults)
    args = ap.parse_args(remaining_argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = SentimentPairDataset(args.train, tokenizer, max_length=args.max_length)
    test_ds = SentimentPairDataset(args.test, tokenizer, max_length=args.max_length)
    print(f"Train examples: {len(train_ds)} (skipped {train_ds.skipped_truncated} truncated-span)")
    print(f"Test examples: {len(test_ds)} (skipped {test_ds.skipped_truncated} truncated-span)")

    train_label_counts = Counter(lbl for _, _, _, lbl in train_ds.examples)
    print("Train sentiment distribution:", {ID2LABEL[k]: v for k, v in train_label_counts.items()})

    if args.loss_type == "logit_adjustment":
        log_priors = compute_log_priors(dict(train_label_counts), len(SENTIMENT_LABELS))
        model = PairClassifier(args.model_name, num_labels=len(SENTIMENT_LABELS),
                                log_priors=log_priors, tau=args.logit_adjustment_tau).to(device)
        print(f"Using logit-adjusted loss (tau={args.logit_adjustment_tau})")
    else:
        weight_by_label = compute_class_weights(
            {ID2LABEL[k]: v for k, v in train_label_counts.items()}, len(SENTIMENT_LABELS),
            cap=args.class_weight_cap
        )
        class_weights = [weight_by_label.get(ID2LABEL[i], args.class_weight_cap) for i in range(len(SENTIMENT_LABELS))]
        model = PairClassifier(args.model_name, num_labels=len(SENTIMENT_LABELS),
                                class_weights=class_weights).to(device)
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

        metrics = evaluate(model, test_ds, device, ID2LABEL)
        neutral_f1 = metrics["per_sentiment"].get("NEUTRAL", {}).get("f1", 0.0)
        print(f"\n[epoch {epoch+1}] accuracy={metrics['accuracy']:.4f} "
              f"macro_f1={metrics['macro_f1_present_categories']:.4f} "
              f"NEUTRAL_f1={neutral_f1:.4f}  <- watch this one specifically")

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
