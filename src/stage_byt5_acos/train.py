"""
Training Script for ByT5 End-to-End Amharic ACOS Quadruple Extraction
======================================================================
Trains google/byt5-base with byte-level tokenization, gradient accumulation,
Adafactor/AdamW optimization, mixed precision, and full state checkpointing.

Usage:
    PYTHONPATH=src/stage_byt5_acos python src/stage_byt5_acos/train.py \
        --config configs/stage_byt5_acos.yaml \
        --output_dir results/stage_byt5_acos/byt5_base_run1
"""
import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from dataset import ByT5ACOSDataset, collate_fn
from evaluate import evaluate_byt5_model, print_metrics_table
from model import ByT5ACOSModel, build_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ByT5 for Amharic ACOS Quadruple Extraction.")
    parser.add_argument("--config", default="configs/stage_byt5_acos.yaml", help="Path to config YAML")
    parser.add_argument("--output_dir", default=None, help="Output directory for checkpoints and logs")
    parser.add_argument("--model_name", default=None, help="HuggingFace model name (e.g. google/byt5-base)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per forward step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--optimizer", default=None, choices=["adafactor", "adamw"], help="Optimizer type")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for validation generation")
    parser.add_argument("--eval_beams", type=int, default=1, help="Beam count for validation generation")
    parser.add_argument("--resume", action="store_true", help="Resume from last_checkpoint.pt or best_model.pt")
    parser.add_argument("--resume_from", default=None, help="Explicit path to checkpoint file to resume from")
    parser.add_argument("--start_epoch", type=int, default=None, help="Explicit start epoch")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable automatic mixed precision")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16", help="Disable mixed precision")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = args.model_name or cfg.get("model_name", "google/byt5-base")
    output_dir = args.output_dir or cfg.get("output_dir", "results/stage_byt5_acos/byt5_base_run1")
    os.makedirs(output_dir, exist_ok=True)

    training_cfg = cfg.get("training", {})
    batch_size = args.batch_size or training_cfg.get("batch_size", 4)
    grad_accum_steps = args.gradient_accumulation_steps or training_cfg.get("gradient_accumulation_steps", 4)
    epochs = args.epochs or training_cfg.get("epochs", 10)
    lr = args.lr or float(training_cfg.get("lr", 1e-4))
    optimizer_type = args.optimizer or training_cfg.get("optimizer", "adafactor")
    weight_decay = float(training_cfg.get("weight_decay", 0.01))

    model_cfg = cfg.get("model", {})
    max_source_length = model_cfg.get("max_source_length", 768)
    max_target_length = model_cfg.get("max_target_length", 384)

    data_cfg = cfg.get("data", {})
    train_path = data_cfg.get("train", "data/prepared/train.jsonl")
    dev_path = data_cfg.get("dev", "data/prepared/dev.jsonl")
    test_path = data_cfg.get("test", "data/prepared/test.jsonl")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    use_cuda = device.type == "cuda"
    use_fp16 = args.fp16 and use_cuda

    print(f"Device: {device}")
    print(f"Model backbone: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size} (Grad Accum: {grad_accum_steps} -> Effective batch: {batch_size * grad_accum_steps})")
    print(f"Optimizer: {optimizer_type} | LR: {lr} | Mixed Precision (FP16): {use_fp16}")

    # Save resolved arguments
    resolved_args = {
        "model_name": model_name,
        "train_path": train_path,
        "dev_path": dev_path,
        "test_path": test_path,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "epochs": epochs,
        "lr": lr,
        "optimizer": optimizer_type,
        "weight_decay": weight_decay,
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "fp16": use_fp16,
    }
    with open(os.path.join(output_dir, "resolved_args.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_args, f, indent=2)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading datasets...")
    train_dataset = ByT5ACOSDataset(
        train_path, tokenizer, max_source_length=max_source_length, max_target_length=max_target_length
    )
    dev_dataset = ByT5ACOSDataset(
        dev_path, tokenizer, max_source_length=max_source_length, max_target_length=max_target_length
    )
    test_dataset = ByT5ACOSDataset(
        test_path, tokenizer, max_source_length=max_source_length, max_target_length=max_target_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Dev samples  : {len(dev_dataset):,}")
    print(f"Test samples : {len(test_dataset):,}")

    print("\nInitializing ByT5 model...")
    model = ByT5ACOSModel(model_name=model_name)
    model.to(device)

    optimizer = build_optimizer(model, optimizer_type=optimizer_type, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda') if use_fp16 else None

    best_dev_f1 = -1.0
    best_checkpoint_path = os.path.join(output_dir, "best_model.pt")
    last_checkpoint_path = os.path.join(output_dir, "last_checkpoint.pt")
    start_epoch = 1

    # Checkpoint Resumption
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
            start_epoch = loaded.get("epoch", 0) + 1
            best_dev_f1 = loaded.get("best_dev_f1", -1.0)
            print(f"  ✓ Restored full checkpoint state. Resuming at epoch {start_epoch}/{epochs} (Prior Best Dev F1: {best_dev_f1*100:.2f}%)")
        else:
            model.load_state_dict(loaded)
            best_dev_metrics_file = os.path.join(output_dir, "best_dev_metrics.json")
            if os.path.exists(best_dev_metrics_file):
                with open(best_dev_metrics_file, encoding="utf-8") as f:
                    saved_metrics = json.load(f)
                    best_dev_f1 = saved_metrics.get("full_quad", {}).get("f1", -1.0)
            if args.start_epoch is not None:
                start_epoch = args.start_epoch
            print(f"  ✓ Restored model weights. Resuming from epoch {start_epoch}/{epochs} (Prior Best Dev F1: {best_dev_f1*100:.2f}%)")

    if start_epoch > epochs:
        print(f"\nTraining already completed through epoch {epochs}! Proceeding to final test evaluation...")
    else:
        print(f"\nStarting ByT5 training from epoch {start_epoch} to {epochs}...")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        step_in_epoch = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_fp16:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / grad_accum_steps
                loss.backward()

            total_epoch_loss += loss.item() * grad_accum_steps
            step_in_epoch += 1

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        avg_train_loss = total_epoch_loss / max(1, step_in_epoch)
        print(f"\nEpoch {epoch} complete | Average Train Loss: {avg_train_loss:.4f}")

        # Evaluation on Dev Set
        print("Evaluating on Dev split...")
        dev_metrics = evaluate_byt5_model(
            model=model,
            dataloader=dev_loader,
            tokenizer=tokenizer,
            device=device,
            max_target_length=max_target_length,
            num_beams=args.eval_beams,
        )
        print_metrics_table(dev_metrics, title=f"Dev Metrics (Epoch {epoch})")

        dev_f1 = dev_metrics["full_quad"]["f1"]
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict(), best_checkpoint_path)
            with open(os.path.join(output_dir, "best_dev_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, indent=2)
            print(f"  ★ New best Dev Full Quad F1: {dev_f1*100:.2f}% -> Checkpoint saved!")

        # Save last checkpoint for crash recovery
        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dev_f1": best_dev_f1,
            "dev_metrics": dev_metrics,
        }
        torch.save(last_ckpt, last_checkpoint_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final Test Split Evaluation
    print("\n" + "=" * 80)
    print("FINAL TEST SPLIT EVALUATION USING BEST CHECKPOINT")
    print("=" * 80)
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best weights from {best_checkpoint_path}...")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    test_metrics = evaluate_byt5_model(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_target_length=max_target_length,
        num_beams=args.eval_beams,
    )
    print_metrics_table(test_metrics, title="Final Test Evaluation Results")

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved final test metrics to: {os.path.join(output_dir, 'test_metrics.json')}")


if __name__ == "__main__":
    main()
