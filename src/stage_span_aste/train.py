"""
Training Script for Span-ASTE (ACL 2021)
========================================
Trains the Span-ASTE model on Amharic ACOS explicit-both pairs/triplets:
  - AdamW optimizer with differential learning rates (5e-5 for encoder, 1e-3 for heads)
  - Linear warmup for 10% steps + linear decay
  - Evaluates ASTE, AOPE, ATE, and OTE on dev set every epoch
  - Saves best checkpoint and test results to output directory
"""
import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from dataset import SpanASTEDataset, collate_fn
from evaluation import evaluate_batch_predictions, summarize_metrics
from model import SpanASTEModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train Span-ASTE model.")
    parser.add_argument("--config", default="configs/stage_span_aste.yaml", help="Path to YAML config")
    parser.add_argument("--train", default=None, help="Train JSONL path")
    parser.add_argument("--dev", default=None, help="Dev JSONL path")
    parser.add_argument("--test", default=None, help="Test JSONL path")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr_transformer", type=float, default=None)
    parser.add_argument("--lr_head", type=float, default=None)
    parser.add_argument("--max_span_length", type=int, default=None)
    parser.add_argument("--pruning_ratio", type=float, default=None)
    parser.add_argument("--use_biaffine", action="store_true", default=None, help="Use deep biaffine relation classifier")
    parser.add_argument("--no_biaffine", action="store_false", dest="use_biaffine", help="Disable biaffine classifier, use legacy MLP")
    parser.add_argument("--biaffine_dim", type=int, default=None, help="Biaffine projection dimension")
    parser.add_argument("--use_span_sync", action="store_true", default=None, help="Enable cross-span contextual synchronization")
    parser.add_argument("--no_span_sync", action="store_false", dest="use_span_sync", help="Disable cross-span synchronization")
    parser.add_argument("--use_span_mean_pooling", action="store_true", default=None, help="Enable span mean pooling")
    parser.add_argument("--no_span_mean_pooling", action="store_false", dest="use_span_mean_pooling", help="Disable span mean pooling")
    parser.add_argument("--use_focal_loss", action="store_true", default=None, help="Use Focal Loss for relations")
    parser.add_argument("--no_focal_loss", action="store_false", dest="use_focal_loss", help="Use standard Cross Entropy for relations")
    parser.add_argument("--focal_gamma", type=float, default=None, help="Focal loss gamma parameter")
    parser.add_argument("--relation_loss_weights", type=float, nargs="+", default=None, help="Class weights for [INVALID, POS, NEG, NEU]")
    parser.add_argument("--resume", action="store_true", help="Resume training from last_checkpoint.pt or best_model.pt in output_dir")
    parser.add_argument("--resume_from", default=None, help="Explicit path to checkpoint file to resume from")
    parser.add_argument("--start_epoch", type=int, default=None, help="Explicit start epoch (e.g. 8 when resuming from weights-only checkpoint)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_evaluation(model, dataloader, device) -> dict:
    """Evaluates model over a DataLoader and computes full metrics."""
    model.eval()
    accumulated_counts = {
        "aste": {"tp": 0, "fp": 0, "fn": 0},
        "aope": {"tp": 0, "fp": 0, "fn": 0},
        "ate": {"tp": 0, "fp": 0, "fn": 0},
        "ote": {"tp": 0, "fp": 0, "fn": 0},
        "candidate_ceiling": {"reachable": 0, "total_gold_pairs": 0},
    }

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            subword_to_word = batch["subword_to_word"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                subword_to_word=subword_to_word,
                seq_spans=batch["seq_spans"],
                gold_mention_labels=None,  # No gold labels during inference evaluation
                gold_pairs=None,
                num_words=batch["num_words"],
            )

            batch_counts = evaluate_batch_predictions(
                batch_outputs=out["batch_outputs"],
                raw_quads=batch["raw_quads"],
            )

            for task in ("aste", "aope", "ate", "ote"):
                for k in ("tp", "fp", "fn"):
                    accumulated_counts[task][k] += batch_counts[task][k]
            accumulated_counts["candidate_ceiling"]["reachable"] += batch_counts["candidate_ceiling"]["reachable"]
            accumulated_counts["candidate_ceiling"]["total_gold_pairs"] += batch_counts["candidate_ceiling"]["total_gold_pairs"]

    return summarize_metrics(accumulated_counts)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve settings from CLI or config
    model_name = args.model_name or cfg.get("model_name", "Davlan/afro-xlmr-base")
    output_dir = args.output_dir or cfg.get("output_dir", "results/stage_span_aste/afroxlmr_run1")
    os.makedirs(output_dir, exist_ok=True)

    data_cfg = cfg.get("data", {})
    train_path = args.train or data_cfg.get("train", "data/prepared_explicit/train.jsonl")
    dev_path = args.dev or data_cfg.get("dev", "data/prepared_explicit/dev.jsonl")
    test_path = args.test or data_cfg.get("test", "data/prepared_explicit/test.jsonl")

    training_cfg = cfg.get("training", {})
    batch_size = args.batch_size or training_cfg.get("batch_size", 4)
    epochs = args.epochs or training_cfg.get("epochs", 10)
    lr_transformer = args.lr_transformer or training_cfg.get("lr_transformer", 5e-5)
    lr_head = args.lr_head or training_cfg.get("lr_head", 1e-3)
    weight_decay = training_cfg.get("weight_decay", 1e-2)
    warmup_ratio = training_cfg.get("warmup_ratio", 0.1)

    model_cfg = cfg.get("model", {})
    max_length = model_cfg.get("max_length", 256)
    max_words = model_cfg.get("max_words", 128)
    max_span_length = args.max_span_length or model_cfg.get("max_span_length", 8)
    pruning_ratio = args.pruning_ratio or model_cfg.get("pruning_ratio", 0.5)
    width_dim = model_cfg.get("width_dim", 20)
    distance_dim = model_cfg.get("distance_dim", 128)
    hidden_dim = model_cfg.get("hidden_dim", 150)
    dropout = model_cfg.get("dropout", 0.4)
    use_biaffine = args.use_biaffine if args.use_biaffine is not None else model_cfg.get("use_biaffine", True)
    biaffine_dim = args.biaffine_dim or model_cfg.get("biaffine_dim", 256)
    use_span_sync = args.use_span_sync if args.use_span_sync is not None else model_cfg.get("use_span_sync", True)
    use_span_mean_pooling = args.use_span_mean_pooling if args.use_span_mean_pooling is not None else model_cfg.get("use_span_mean_pooling", True)
    use_focal_loss = args.use_focal_loss if args.use_focal_loss is not None else model_cfg.get("use_focal_loss", False)
    focal_gamma = args.focal_gamma if args.focal_gamma is not None else model_cfg.get("focal_gamma", 2.0)
    relation_loss_weights = args.relation_loss_weights if args.relation_loss_weights is not None else model_cfg.get("relation_loss_weights", None)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"Model backbone: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Biaffine relation scoring: {use_biaffine} (dim: {biaffine_dim}, span_sync: {use_span_sync})")
    print(f"Span mean pooling: {use_span_mean_pooling}")
    print(f"Relation loss mode: {'Focal Loss (gamma=' + str(focal_gamma) + ')' if use_focal_loss else 'Cross-Entropy'} | Weights: {relation_loss_weights}")

    # Save resolved arguments
    resolved_args = {
        "model_name": model_name,
        "train_path": train_path,
        "dev_path": dev_path,
        "test_path": test_path,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr_transformer": lr_transformer,
        "lr_head": lr_head,
        "max_span_length": max_span_length,
        "pruning_ratio": pruning_ratio,
        "width_dim": width_dim,
        "distance_dim": distance_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "use_biaffine": use_biaffine,
        "biaffine_dim": biaffine_dim,
        "use_span_sync": use_span_sync,
        "use_span_mean_pooling": use_span_mean_pooling,
        "use_focal_loss": use_focal_loss,
        "focal_gamma": focal_gamma,
        "relation_loss_weights": relation_loss_weights,
    }
    with open(os.path.join(output_dir, "resolved_args.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_args, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading datasets...")
    train_dataset = SpanASTEDataset(
        train_path,
        tokenizer,
        max_length=max_length,
        max_words=max_words,
        max_span_length=max_span_length,
    )
    dev_dataset = SpanASTEDataset(
        dev_path,
        tokenizer,
        max_length=max_length,
        max_words=max_words,
        max_span_length=max_span_length,
    )
    test_dataset = SpanASTEDataset(
        test_path,
        tokenizer,
        max_length=max_length,
        max_words=max_words,
        max_span_length=max_span_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Dev samples  : {len(dev_dataset):,}")
    print(f"Test samples : {len(test_dataset):,}")

    model = SpanASTEModel(
        model_name=model_name,
        max_span_length=max_span_length,
        pruning_ratio=pruning_ratio,
        width_dim=width_dim,
        distance_dim=distance_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_biaffine=use_biaffine,
        biaffine_dim=biaffine_dim,
        use_span_sync=use_span_sync,
        use_span_mean_pooling=use_span_mean_pooling,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        relation_loss_weights=relation_loss_weights,
    )
    model.to(device)

    # Differential learning rates matching Section 3.2 of the paper:
    # 5e-5 for transformer with weight decay 1e-2; 1e-3 for task heads with 0 weight decay
    encoder_params = list(model.encoder.parameters())
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.")
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_transformer, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": 0.0},
        ]
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_dev_f1 = -1.0
    best_checkpoint_path = os.path.join(output_dir, "best_model.pt")
    last_checkpoint_path = os.path.join(output_dir, "last_checkpoint.pt")
    start_epoch = 1

    checkpoint_to_resume = None
    if args.resume_from:
        checkpoint_to_resume = args.resume_from
    elif args.resume:
        if os.path.exists(last_checkpoint_path):
            checkpoint_to_resume = last_checkpoint_path
        elif os.path.exists(best_checkpoint_path):
            checkpoint_to_resume = best_checkpoint_path

    if checkpoint_to_resume and os.path.exists(checkpoint_to_resume):
        print(f"\nResuming training from: {checkpoint_to_resume}")
        loaded = torch.load(checkpoint_to_resume, map_location=device)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            model.load_state_dict(loaded["model_state_dict"])
            if "optimizer_state_dict" in loaded:
                optimizer.load_state_dict(loaded["optimizer_state_dict"])
            if "scheduler_state_dict" in loaded:
                scheduler.load_state_dict(loaded["scheduler_state_dict"])
            start_epoch = loaded.get("epoch", 0) + 1
            best_dev_f1 = loaded.get("best_dev_f1", -1.0)
            print(f"  ✓ Restored full checkpoint state. Resuming at epoch {start_epoch}/{epochs} (Prior Best Dev F1: {best_dev_f1*100:.2f}%)")
        else:
            model.load_state_dict(loaded)
            best_dev_metrics_file = os.path.join(output_dir, "best_dev_metrics.json")
            if os.path.exists(best_dev_metrics_file):
                with open(best_dev_metrics_file, encoding="utf-8") as f:
                    saved_metrics = json.load(f)
                    best_dev_f1 = saved_metrics.get("aope", {}).get("f1", -1.0)
            if args.start_epoch is not None:
                start_epoch = args.start_epoch
            print(f"  ✓ Restored model weights. Starting from epoch {start_epoch}/{epochs} (Prior Best Dev F1: {best_dev_f1*100:.2f}%)")
            if start_epoch > 1:
                steps_to_advance = (start_epoch - 1) * len(train_loader)
                for _ in range(steps_to_advance):
                    scheduler.step()
                print(f"  ✓ Fast-forwarded learning rate scheduler by {steps_to_advance} steps.")

    if start_epoch > epochs:
        print(f"\nTraining already completed through epoch {epochs}! Proceeding directly to final test evaluation...")
    else:
        print(f"\nStarting training from epoch {start_epoch} to {epochs}...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_m_loss = 0.0
        train_r_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            subword_to_word = batch["subword_to_word"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                subword_to_word=subword_to_word,
                seq_spans=batch["seq_spans"],
                gold_mention_labels=batch["gold_mention_labels"],
                gold_pairs=batch["gold_pairs"],
                num_words=batch["num_words"],
            )

            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_m_loss += out["mention_loss"].item()
            train_r_loss += out["relation_loss"].item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "m_loss": f"{out['mention_loss'].item():.4f}",
                "r_loss": f"{out['relation_loss'].item():.4f}",
            })

        avg_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch} complete | Train Loss: {avg_loss:.4f} (Mention: {train_m_loss/len(train_loader):.4f}, Relation: {train_r_loss/len(train_loader):.4f})")

        # Evaluate on dev set
        print("Evaluating on Dev split...")
        dev_metrics = run_evaluation(model, dev_loader, device)
        aope_f1 = dev_metrics["aope"]["f1"]
        aste_f1 = dev_metrics["aste"]["f1"]
        print(f"  Dev AOPE: P={dev_metrics['aope']['precision']*100:.2f}%, R={dev_metrics['aope']['recall']*100:.2f}%, F1={aope_f1*100:.2f}%")
        print(f"  Dev ASTE: P={dev_metrics['aste']['precision']*100:.2f}%, R={dev_metrics['aste']['recall']*100:.2f}%, F1={aste_f1*100:.2f}%")
        print(f"  Dev ATE : F1={dev_metrics['ate']['f1']*100:.2f}% | Dev OTE: F1={dev_metrics['ote']['f1']*100:.2f}%")
        print(f"  Dev Candidate Ceiling R={dev_metrics['candidate_ceiling_recall']*100:.2f}% | Conversion={dev_metrics['candidate_conversion_efficiency']*100:.2f}%")

        # Save best checkpoint (based on AOPE pair F1)
        if aope_f1 > best_dev_f1:
            best_dev_f1 = aope_f1
            torch.save(model.state_dict(), best_checkpoint_path)
            with open(os.path.join(output_dir, "best_dev_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, indent=2)
            print(f"  ★ New best Dev AOPE F1: {aope_f1*100:.2f}% -> Checkpoint saved!")

        # Always save last_checkpoint.pt with full training state for seamless crash resumption
        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dev_f1": best_dev_f1,
            "dev_metrics": dev_metrics,
        }
        torch.save(last_ckpt, last_checkpoint_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final evaluation on Test split using best checkpoint
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION USING BEST CHECKPOINT")
    print("=" * 80)
    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    test_metrics = run_evaluation(model, test_loader, device)
    with open(os.path.join(output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Test AOPE Pair Extraction: P={test_metrics['aope']['precision']*100:.2f}%, R={test_metrics['aope']['recall']*100:.2f}%, F1={test_metrics['aope']['f1']*100:.2f}%")
    print(f"Test ASTE Triplet Extraction: P={test_metrics['aste']['precision']*100:.2f}%, R={test_metrics['aste']['recall']*100:.2f}%, F1={test_metrics['aste']['f1']*100:.2f}%")
    print(f"Test ATE Aspect Extraction: P={test_metrics['ate']['precision']*100:.2f}%, R={test_metrics['ate']['recall']*100:.2f}%, F1={test_metrics['ate']['f1']*100:.2f}%")
    print(f"Test OTE Opinion Extraction: P={test_metrics['ote']['precision']*100:.2f}%, R={test_metrics['ote']['recall']*100:.2f}%, F1={test_metrics['ote']['f1']*100:.2f}%")
    print(f"Test Candidate Ceiling Recall: {test_metrics['candidate_ceiling_recall']*100:.2f}%")
    print(f"Test Candidate Conversion Efficiency: {test_metrics['candidate_conversion_efficiency']*100:.2f}%")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
