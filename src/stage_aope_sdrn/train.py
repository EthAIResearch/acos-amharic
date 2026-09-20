"""
Training + evaluation for the joint SDRN-style AOPE model. Replaces this
project's Stage 1 (tagging) + Stage 2 (heuristic pairing) as a single
model -- see docs/stage_aope_sdrn_analysis.md for the rationale.

Usage:
    python train.py --config ../../configs/stage_aope_sdrn.yaml

Evaluates aspect-opinion PAIR F1 directly (not just span F1) -- this is
the number to compare against the old pipeline's Stage 1 span F1 -> Stage 2
heuristic pairing F1 combination, and the one that should improve on your
current 14% end-to-end result once this feeds into Stage 3/4/5/6.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

import numpy as np
import torch
import yaml
from align import decode_subword_predictions
from bio_labels import decode_bio_spans
from dataset import JointAOPEDataset, collate_fn
from model import JointAOPESDRN
from relation_utils import (
    compute_prf,
    correlation_degree,
    explicit_pairs,
    sweep_thresholds,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config_defaults(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    flat = {"model_name": cfg.get("model_name"), "output_dir": cfg.get("output_dir"),
            "num_recurrent_steps": cfg.get("num_recurrent_steps"),
            "relation_threshold": cfg.get("relation_threshold"),
            "pair_accept_threshold": cfg.get("pair_accept_threshold")}
    data = cfg.get("data", {})
    flat.update({"train": data.get("train"), "test": data.get("test"), "max_length": data.get("max_length")})
    training = cfg.get("training", {})
    flat.update({"epochs": training.get("epochs"), "batch_size": training.get("batch_size"),
                 "lr": training.get("lr"), "warmup_ratio": training.get("warmup_ratio"),
                 "seed": training.get("seed"), "weight_decay": training.get("weight_decay"),
                 "fp16": training.get("fp16"),
                 "bio_class_weights": training.get("bio_class_weights"),
                 "span_loss_weight": training.get("span_loss_weight")})
    return {k: v for k, v in flat.items() if v is not None}



@torch.no_grad()
def evaluate(
    model,
    dataset,
    tokenizer,
    device,
    pair_accept_threshold,
    batch_size=16,
    return_candidates=False,
):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    all_pred_aspects, all_gold_aspects = [], []
    all_pred_opinions, all_gold_opinions = [], []
    candidates_with_scores_per_ex = []
    all_gold_pairs = []
    rec_idx = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        a_pred_ids = out["aspect_logits"].argmax(-1).cpu().tolist()
        o_pred_ids = out["opinion_logits"].argmax(-1).cpu().tolist()
        rel_scores = torch.sigmoid(out["relation_logits"]).cpu().tolist()

        bsz = input_ids.size(0)
        for i in range(bsz):
            rec = dataset.records[rec_idx]
            rec_idx += 1
            enc = tokenizer(
                rec["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=dataset.max_length,
            )
            word_ids = enc.word_ids(batch_index=0)

            a_word_tags = decode_subword_predictions(word_ids, a_pred_ids[i][:len(word_ids)])
            o_word_tags = decode_subword_predictions(word_ids, o_pred_ids[i][:len(word_ids)])
            a_spans = decode_bio_spans(a_word_tags)
            o_spans = decode_bio_spans(o_word_tags)

            g_pairs = explicit_pairs(rec["quads"])
            g_aspects = [p[0] for p in g_pairs]
            g_opinions = [p[1] for p in g_pairs]

            all_pred_aspects.append(a_spans)
            all_gold_aspects.append(g_aspects)
            all_pred_opinions.append(o_spans)
            all_gold_opinions.append(g_opinions)
            all_gold_pairs.append(g_pairs)

            # Map subword-level relation matrix back to words
            first_subword_of_word = {}
            for si, wid in enumerate(word_ids):
                if wid is not None and wid not in first_subword_of_word:
                    first_subword_of_word[wid] = si
            n_words = len(rec["tokens"])
            word_rel = [[0.0] * n_words for _ in range(n_words)]
            for wi in range(n_words):
                for wj in range(n_words):
                    si, sj = first_subword_of_word.get(wi), first_subword_of_word.get(wj)
                    if (
                        si is not None
                        and sj is not None
                        and si < len(rel_scores[i])
                        and sj < len(rel_scores[i])
                    ):
                        word_rel[wi][wj] = rel_scores[i][si][sj]

            ex_candidates = []
            for a_span in a_spans:
                for o_span in o_spans:
                    score = correlation_degree(word_rel, a_span, o_span)
                    ex_candidates.append(((a_span, o_span), score))
            candidates_with_scores_per_ex.append(ex_candidates)

    aspect_prf = compute_prf(all_pred_aspects, all_gold_aspects)
    opinion_prf = compute_prf(all_pred_opinions, all_gold_opinions)

    accepted_pairs_per_ex = [
        [pair for pair, score in ex_candidates if score >= pair_accept_threshold]
        for ex_candidates in candidates_with_scores_per_ex
    ]
    pair_metrics = compute_prf(accepted_pairs_per_ex, all_gold_pairs)

    all_candidates_per_ex = [
        [pair for pair, _ in ex_candidates]
        for ex_candidates in candidates_with_scores_per_ex
    ]
    candidate_ceiling_prf = compute_prf(all_candidates_per_ex, all_gold_pairs)

    metrics = {
        **pair_metrics,
        "aspect": aspect_prf,
        "opinion": opinion_prf,
        "candidate_ceiling_recall": candidate_ceiling_prf["recall"],
    }

    if return_candidates:
        return metrics, candidates_with_scores_per_ex, all_gold_pairs
    return metrics


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, remaining_argv = pre.parse_known_args()
    config_defaults = load_config_defaults(pre_args.config) if pre_args.config else {}

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--train", required="train" not in config_defaults)
    ap.add_argument("--test", required="test" not in config_defaults)
    ap.add_argument("--model_name", default="Davlan/afro-xlmr-base")
    ap.add_argument("--output_dir", default="./stage_aope_sdrn_ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)  # relation matrix cost -> smaller default batch
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.add_argument("--num_recurrent_steps", type=int, default=2,
                     help="SDRN paper finds 2 steps sufficient (Fig. 3b); more steps show diminishing/negative returns.")
    ap.add_argument("--relation_threshold", type=float, default=0.1,
                     help="Beta in RSM (Eq. 12) -- filters weak relation scores during synchronization.")
    ap.add_argument("--pair_accept_threshold", type=float, default=0.5,
                     help="Delta-hat in Eq. 17 -- correlation degree threshold to accept a pair at inference.")
    ap.add_argument("--bio_class_weights", type=float, nargs="+", default=None,
                     help="Weights for BIO classes [O, B, I] to counter 'O' token dominance.")
    ap.add_argument("--span_loss_weight", type=float, default=1.0,
                     help="Scaling factor for span cross-entropy loss relative to relation BCE loss.")
    ap.set_defaults(**config_defaults)
    args = ap.parse_args(remaining_argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = JointAOPEDataset(args.train, tokenizer, max_length=args.max_length)
    test_ds = JointAOPEDataset(args.test, tokenizer, max_length=args.max_length)

    model = JointAOPESDRN(
        args.model_name,
        num_recurrent_steps=args.num_recurrent_steps,
        relation_threshold=args.relation_threshold,
        bio_class_weights=args.bio_class_weights,
        span_loss_weight=args.span_loss_weight,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
    )

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1 = -1.0
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

        metrics = evaluate(model, test_ds, tokenizer, device, args.pair_accept_threshold)
        print(f"\n[epoch {epoch+1}] pair F1={metrics['f1']:.4f} "
              f"(P={metrics['precision']:.4f} R={metrics['recall']:.4f})")
        print(f"           spans: aspect R={metrics['aspect']['recall']:.4f} (F1={metrics['aspect']['f1']:.4f}), "
              f"opinion R={metrics['opinion']['recall']:.4f} (F1={metrics['opinion']['f1']:.4f}), "
              f"pair ceiling R={metrics['candidate_ceiling_recall']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  -> new best (pair F1={best_f1:.4f}), checkpoint saved")

    print(f"\nDone training. Best pair F1 = {best_f1:.4f}. Checkpoint: {args.output_dir}/best_model.pt")

    # Run threshold sweep on best checkpoint to find optimal delta-hat
    print("\nEvaluating correlation degree threshold sweep on best checkpoint...")
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device), strict=False)
        _, candidates, golds = evaluate(
            model, test_ds, tokenizer, device, args.pair_accept_threshold, return_candidates=True
        )
        sweep_data = sweep_thresholds(candidates, golds)
        print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1 Score':>10}")
        print("-" * 50)
        for row in sweep_data["sweep"]:
            is_best = " *" if row["threshold"] == round(sweep_data["best_threshold"], 4) else ""
            print(f"{row['threshold']:>10.2f} | {row['precision']:>10.4f} | {row['recall']:>10.4f} | {row['f1']:>10.4f}{is_best}")
        print("-" * 50)
        print(f"Optimal threshold by F1: delta-hat = {sweep_data['best_threshold']:.2f} "
              f"(F1={sweep_data['best_metrics']['f1']:.4f}, R={sweep_data['best_metrics']['recall']:.4f})")

        sweep_file = os.path.join(args.output_dir, "threshold_sweep.json")
        with open(sweep_file, "w", encoding="utf-8") as f:
            json.dump(sweep_data, f, indent=2)
        print(f"Saved threshold sweep results to: {sweep_file}")


if __name__ == "__main__":
    main()
