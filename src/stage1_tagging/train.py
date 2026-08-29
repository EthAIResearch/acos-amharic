"""
Stage 1 training: joint Aspect Term Extraction + Opinion Term Extraction.

Usage (in your GPU environment):
    pip install -r requirements.txt
    python train.py --config ../../configs/stage1_afroxlmr.yaml
    python train.py --config ../../configs/stage1_bertsmall.yaml

    # Or without a config file, plain CLI flags (config values above are just
    # defaults for these same flags -- any flag passed on the command line
    # overrides the config):
    python train.py \
        --train ../../data/prepared/train.jsonl \
        --test ../../data/prepared/test.jsonl \
        --model_name Davlan/afro-xlmr-base \
        --output_dir ../../results/stage1/afroxlmr_base_run1 \
        --epochs 8 --batch_size 16 --lr 3e-5
"""
import argparse
import json
import os
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset_tagging import TaggingDataset, collate_fn
from model_tagging import JointTaggingModel
from bio_labels import decode_bio_spans
from align import decode_subword_predictions


def span_prf(pred_spans_per_ex, gold_spans_per_ex):
    tp = fp = fn = 0
    for preds, golds in zip(pred_spans_per_ex, gold_spans_per_ex):
        preds_set, golds_set = set(preds), set(golds)
        tp += len(preds_set & golds_set)
        fp += len(preds_set - golds_set)
        fn += len(golds_set - preds_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def evaluate(model, dataset, tokenizer, device, batch_size=32):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_a_pred, all_a_gold, all_o_pred, all_o_gold = [], [], [], []
    rec_idx = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        a_pred_ids = out["aspect_logits"].argmax(-1).cpu().tolist()
        o_pred_ids = out["opinion_logits"].argmax(-1).cpu().tolist()

        bsz = input_ids.size(0)
        for i in range(bsz):
            rec = dataset.records[rec_idx]
            rec_idx += 1
            enc = tokenizer(rec["tokens"], is_split_into_words=True,
                             truncation=True, max_length=dataset.max_length)
            word_ids = enc.word_ids(batch_index=0)

            a_word_tags = decode_subword_predictions(word_ids, a_pred_ids[i][:len(word_ids)])
            o_word_tags = decode_subword_predictions(word_ids, o_pred_ids[i][:len(word_ids)])

            all_a_pred.append(decode_bio_spans(a_word_tags))
            all_o_pred.append(decode_bio_spans(o_word_tags))
            all_a_gold.append([(q["a_start"], q["a_end"]) for q in rec["quads"] if q["a_start"] != -1])
            all_o_gold.append([(q["o_start"], q["o_end"]) for q in rec["quads"] if q["o_start"] != -1])

    return {
        "aspect": span_prf(all_a_pred, all_a_gold),
        "opinion": span_prf(all_o_pred, all_o_gold),
    }


def load_config_defaults(config_path):
    """Flatten our nested config YAML (model_name / data.* / training.* /
    output_dir) into the flat CLI flag names train.py already uses."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    flat = {}
    flat["model_name"] = cfg.get("model_name")
    flat["output_dir"] = cfg.get("output_dir")
    data = cfg.get("data", {})
    flat["train"] = data.get("train")
    flat["test"] = data.get("test")
    flat["max_length"] = data.get("max_length")
    training = cfg.get("training", {})
    flat["epochs"] = training.get("epochs")
    flat["batch_size"] = training.get("batch_size")
    flat["lr"] = training.get("lr")
    flat["warmup_ratio"] = training.get("warmup_ratio")
    flat["seed"] = training.get("seed")
    return {k: v for k, v in flat.items() if v is not None}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # Pre-parse just --config so its values become argparse defaults, letting
    # any explicit CLI flag still override the config (config values take
    # priority over hardcoded defaults; CLI flags take priority over config).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, remaining_argv = pre.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config) if pre_args.config else {}

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--train", required="train" not in config_defaults)
    ap.add_argument("--test", required="test" not in config_defaults)
    ap.add_argument("--model_name", default="Davlan/afro-xlmr-base")
    ap.add_argument("--output_dir", default="./stage1_tagging_ckpt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.set_defaults(**config_defaults)
    args = ap.parse_args(remaining_argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "resolved_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = TaggingDataset(args.train, tokenizer, max_length=args.max_length)
    test_ds = TaggingDataset(args.test, tokenizer, max_length=args.max_length)

    model = JointTaggingModel(args.model_name).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
    )

    best_f1 = -1.0
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

        metrics = evaluate(model, test_ds, tokenizer, device)
        print(f"\n[epoch {epoch+1}] aspect F1={metrics['aspect']['f1']:.4f} "
              f"(P={metrics['aspect']['precision']:.4f} R={metrics['aspect']['recall']:.4f})  "
              f"opinion F1={metrics['opinion']['f1']:.4f} "
              f"(P={metrics['opinion']['precision']:.4f} R={metrics['opinion']['recall']:.4f})")

        avg_f1 = (metrics["aspect"]["f1"] + metrics["opinion"]["f1"]) / 2
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  -> new best (avg F1={avg_f1:.4f}), checkpoint saved")

    print(f"\nDone. Best avg span F1 = {best_f1:.4f}. Checkpoint: {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
