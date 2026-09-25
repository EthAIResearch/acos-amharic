"""
Evaluation Script for Generative ByT5 ACOS Quadruple Extraction
================================================================
Generates predictions from a trained ByT5 model, parses structured quadruples,
and calculates comprehensive metrics (Full Quad, Explicit-only, Implicit-only,
ASTE Triplet, AOPE Pair, ACSE Triplet).

Usage:
    PYTHONPATH=src/stage_byt5_acos python src/stage_byt5_acos/evaluate.py \
        --checkpoint results/stage_byt5_acos/byt5_base_run1/best_model.pt \
        --config configs/stage_byt5_acos.yaml \
        --split test
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
from constrained_decoder import ACOSByteFSM
from dataset import ByT5ACOSDataset, collate_fn
from linearization import (
    calculate_metrics,
    compute_clitic_normalized_set_prf,
    compute_set_prf,
    parse_target_to_quads,
)
from model import ByT5ACOSModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ByT5 ACOS Quadruple Extraction model.")
    parser.add_argument("--config", default="configs/stage_byt5_acos.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint best_model.pt")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="test", help="Split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam size for generation (1 = greedy, 4 = beam search)")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty for beam search")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty for generation")
    parser.add_argument("--constrained", action="store_true", help="Enable byte-level FSM constrained decoding")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Inference precision")
    parser.add_argument("--output_file", default=None, help="Path to save evaluation metrics JSON")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def evaluate_byt5_model(
    model: ByT5ACOSModel,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_target_length: int = 256,
    num_beams: int = 1,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.2,
    constrained: bool = False,
    precision: str = "bf16",
) -> dict:
    """
    Evaluates ByT5 model over a DataLoader and computes full ACOS metrics,
    including both strict exact match and Amharic clitic-normalized match.
    """
    model.eval()

    counts = {
        "full_quad": {"tp": 0, "fp": 0, "fn": 0},
        "explicit_quad": {"tp": 0, "fp": 0, "fn": 0},
        "implicit_quad": {"tp": 0, "fp": 0, "fn": 0},
        "aope": {"tp": 0, "fp": 0, "fn": 0},
        "aste": {"tp": 0, "fp": 0, "fn": 0},
        "acse": {"tp": 0, "fp": 0, "fn": 0},
        "cat_sent": {"tp": 0, "fp": 0, "fn": 0},
    }

    counts_norm = {
        "full_quad": {"tp": 0, "fp": 0, "fn": 0},
        "explicit_quad": {"tp": 0, "fp": 0, "fn": 0},
        "implicit_quad": {"tp": 0, "fp": 0, "fn": 0},
        "aope": {"tp": 0, "fp": 0, "fn": 0},
        "aste": {"tp": 0, "fp": 0, "fn": 0},
    }

    sample_predictions = []

    logits_processor = [ACOSByteFSM(tokenizer)] if constrained else None

    autocast_ctx = (
        torch.amp.autocast('cuda', dtype=torch.bfloat16)
        if (device.type == "cuda" and precision == "bf16")
        else (
            torch.amp.autocast('cuda', dtype=torch.float16)
            if (device.type == "cuda" and precision == "fp16")
            else torch.nullcontext()
        )
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating ACOS quads (beams={num_beams}, constrained={constrained})"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_length": max_target_length,
                "num_beams": num_beams,
                "length_penalty": length_penalty,
                "repetition_penalty": repetition_penalty,
            }
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor

            with autocast_ctx:
                generated_ids = model.generate(**gen_kwargs)

            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for pred_str, gold_quads, raw_text, gold_target_str in zip(
                pred_texts, batch["gold_quads"], batch["raw_texts"], batch["raw_targets"]
            ):
                pred_quads = parse_target_to_quads(pred_str)

                # ==========================
                # 1. STRICT EXACT MATCH
                # ==========================
                # Full Quadruple (a, c, s, o)
                tp, fp, fn = compute_set_prf(pred_quads, gold_quads)
                counts["full_quad"]["tp"] += tp
                counts["full_quad"]["fp"] += fp
                counts["full_quad"]["fn"] += fn

                # Explicit-only vs Implicit-only Quads
                pred_exp = [q for q in pred_quads if q[0] != "NULL" and q[3] != "NULL"]
                gold_exp = [q for q in gold_quads if q[0] != "NULL" and q[3] != "NULL"]
                tp_e, fp_e, fn_e = compute_set_prf(pred_exp, gold_exp)
                counts["explicit_quad"]["tp"] += tp_e
                counts["explicit_quad"]["fp"] += fp_e
                counts["explicit_quad"]["fn"] += fn_e

                pred_imp = [q for q in pred_quads if q[0] == "NULL" or q[3] == "NULL"]
                gold_imp = [q for q in gold_quads if q[0] == "NULL" or q[3] == "NULL"]
                tp_i, fp_i, fn_i = compute_set_prf(pred_imp, gold_imp)
                counts["implicit_quad"]["tp"] += tp_i
                counts["implicit_quad"]["fp"] += fp_i
                counts["implicit_quad"]["fn"] += fn_i

                # AOPE (a, o)
                pred_aope = [(q[0], q[3]) for q in pred_quads]
                gold_aope = [(q[0], q[3]) for q in gold_quads]
                tp_p, fp_p, fn_p = compute_set_prf(pred_aope, gold_aope)
                counts["aope"]["tp"] += tp_p
                counts["aope"]["fp"] += fp_p
                counts["aope"]["fn"] += fn_p

                # ASTE (a, s, o)
                pred_aste = [(q[0], q[2], q[3]) for q in pred_quads]
                gold_aste = [(q[0], q[2], q[3]) for q in gold_quads]
                tp_t, fp_t, fn_t = compute_set_prf(pred_aste, gold_aste)
                counts["aste"]["tp"] += tp_t
                counts["aste"]["fp"] += fp_t
                counts["aste"]["fn"] += fn_t

                # ACSE (a, c, s)
                pred_acse = [(q[0], q[1], q[2]) for q in pred_quads]
                gold_acse = [(q[0], q[1], q[2]) for q in gold_quads]
                tp_c, fp_c, fn_c = compute_set_prf(pred_acse, gold_acse)
                counts["acse"]["tp"] += tp_c
                counts["acse"]["fp"] += fp_c
                counts["acse"]["fn"] += fn_c

                # Category-Sentiment (c, s)
                pred_cs = [(q[1], q[2]) for q in pred_quads]
                gold_cs = [(q[1], q[2]) for q in gold_quads]
                tp_s, fp_s, fn_s = compute_set_prf(pred_cs, gold_cs)
                counts["cat_sent"]["tp"] += tp_s
                counts["cat_sent"]["fp"] += fp_s
                counts["cat_sent"]["fn"] += fn_s

                # ==========================
                # 2. CLITIC-NORMALIZED MATCH
                # ==========================
                tp_nq, fp_nq, fn_nq = compute_clitic_normalized_set_prf(pred_quads, gold_quads)
                counts_norm["full_quad"]["tp"] += tp_nq
                counts_norm["full_quad"]["fp"] += fp_nq
                counts_norm["full_quad"]["fn"] += fn_nq

                tp_nexp, fp_nexp, fn_nexp = compute_clitic_normalized_set_prf(pred_exp, gold_exp)
                counts_norm["explicit_quad"]["tp"] += tp_nexp
                counts_norm["explicit_quad"]["fp"] += fp_nexp
                counts_norm["explicit_quad"]["fn"] += fn_nexp

                tp_nimp, fp_nimp, fn_nimp = compute_clitic_normalized_set_prf(pred_imp, gold_imp)
                counts_norm["implicit_quad"]["tp"] += tp_nimp
                counts_norm["implicit_quad"]["fp"] += fp_nimp
                counts_norm["implicit_quad"]["fn"] += fn_nimp

                tp_np, fp_np, fn_np = compute_clitic_normalized_set_prf(pred_aope, gold_aope)
                counts_norm["aope"]["tp"] += tp_np
                counts_norm["aope"]["fp"] += fp_np
                counts_norm["aope"]["fn"] += fn_np

                tp_nt, fp_nt, fn_nt = compute_clitic_normalized_set_prf(pred_aste, gold_aste)
                counts_norm["aste"]["tp"] += tp_nt
                counts_norm["aste"]["fp"] += fp_nt
                counts_norm["aste"]["fn"] += fn_nt

                if len(sample_predictions) < 5:
                    sample_predictions.append({
                        "text": raw_text,
                        "pred_string": pred_str,
                        "gold_string": gold_target_str,
                        "pred_quads": pred_quads,
                        "gold_quads": gold_quads,
                    })

    strict_metrics = {
        task: calculate_metrics(v["tp"], v["fp"], v["fn"])
        for task, v in counts.items()
    }
    clitic_norm_metrics = {
        task: calculate_metrics(v["tp"], v["fp"], v["fn"])
        for task, v in counts_norm.items()
    }

    # Combined dictionary (strict metrics top-level for backward compatibility)
    metrics = dict(strict_metrics)
    metrics["clitic_normalized"] = clitic_norm_metrics
    metrics["sample_predictions"] = sample_predictions
    metrics["generation_config"] = {
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "repetition_penalty": repetition_penalty,
        "constrained": constrained,
    }
    return metrics


def print_metrics_table(metrics: dict, title: str = "ByT5 ACOS Evaluation Results"):
    print("\n" + "=" * 80)
    print(f"  {title} — Strict Exact Match")
    print("=" * 80)
    print(f"  {'Task / Phenotype':<22} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print("-" * 80)

    task_labels = {
        "full_quad": "Full Quad (A,C,S,O)",
        "explicit_quad": "Explicit-only Quads",
        "implicit_quad": "Implicit-only Quads",
        "aste": "ASTE (A, S, O)",
        "aope": "AOPE (A, O)",
        "acse": "ACSE (A, C, S)",
        "cat_sent": "Pair (Category, Sent)",
    }

    for task_key, label in task_labels.items():
        if task_key in metrics and isinstance(metrics[task_key], dict) and "precision" in metrics[task_key]:
            m = metrics[task_key]
            p = m["precision"] * 100
            r = m["recall"] * 100
            f1 = m["f1"] * 100
            print(f"  {label:<22} | {p:<9.2f}% | {r:<9.2f}% | {f1:<9.2f}% | {m['tp']:<6} | {m['fp']:<6} | {m['fn']:<6}")

    print("=" * 80)

    # Print Clitic-Normalized Table if present
    if "clitic_normalized" in metrics:
        norm_m = metrics["clitic_normalized"]
        print("\n" + "=" * 80)
        print(f"  {title} — Amharic Clitic-Normalized (Morphological Invariance)")
        print("=" * 80)
        print(f"  {'Task / Phenotype':<22} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
        print("-" * 80)
        for task_key, label in task_labels.items():
            if task_key in norm_m:
                m = norm_m[task_key]
                p = m["precision"] * 100
                r = m["recall"] * 100
                f1 = m["f1"] * 100
                print(f"  {label:<22} | {p:<9.2f}% | {r:<9.2f}% | {f1:<9.2f}% | {m['tp']:<6} | {m['fp']:<6} | {m['fn']:<6}")
        print("=" * 80)



def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg.get("model_name", "google/byt5-base")
    output_dir = cfg.get("output_dir", "results/stage_byt5_acos/byt5_base_run1")
    checkpoint_path = args.checkpoint or os.path.join(output_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if args.precision == "auto":
        if device.type == "cuda":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
        else:
            precision = "fp32"
    else:
        if device.type != "cuda" and args.precision in {"bf16", "fp16"}:
            precision = "fp32"
        else:
            precision = args.precision

    eval_cfg = cfg.get("evaluation", {})
    num_beams = args.num_beams if args.num_beams > 1 else eval_cfg.get("num_beams", args.num_beams)
    length_penalty = args.length_penalty if args.length_penalty != 1.0 else eval_cfg.get("length_penalty", args.length_penalty)
    repetition_penalty = args.repetition_penalty if args.repetition_penalty != 1.2 else eval_cfg.get("repetition_penalty", args.repetition_penalty)
    constrained = args.constrained or eval_cfg.get("constrained", False)

    print(f"Using device: {device} | Precision: {precision}")
    print(f"Generation settings: beams={num_beams} | length_penalty={length_penalty} | repetition_penalty={repetition_penalty} | constrained={constrained}")

    model_cfg = cfg.get("model", {})
    max_source_length = model_cfg.get("max_source_length", 768)
    max_target_length = model_cfg.get("max_target_length", 256)

    data_cfg = cfg.get("data", {})
    splits_to_evaluate = ["dev", "test"] if args.split == "both" else [args.split]

    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model weights from {checkpoint_path}...")
    model = ByT5ACOSModel(model_name=model_name)
    loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        model.load_state_dict(loaded["model_state_dict"])
    else:
        model.load_state_dict(loaded)
    model.to(device)

    for split in splits_to_evaluate:
        data_path = data_cfg.get(split, f"data/prepared/{split}.jsonl")
        print(f"\nLoading {split.upper()} dataset from {data_path}...")
        dataset = ByT5ACOSDataset(
            data_path,
            tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        metrics = evaluate_byt5_model(
            model=model,
            dataloader=loader,
            tokenizer=tokenizer,
            device=device,
            max_target_length=max_target_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            constrained=constrained,
            precision=precision,
        )

        title_suffix = " (Constrained FSM)" if constrained else " (Standard)"
        print_metrics_table(metrics, title=f"ByT5 ACOS Evaluation ({split.upper()} Split){title_suffix}")

        if args.output_file:
            out_path = args.output_file
        else:
            suffix = "_constrained" if constrained else ""
            if num_beams > 1:
                suffix += f"_beam{num_beams}"
            out_path = os.path.join(output_dir, f"{split}{suffix}_metrics.json")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to: {out_path}")


if __name__ == "__main__":
    main()
