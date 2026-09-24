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
from dataset import ByT5ACOSDataset, collate_fn
from linearization import calculate_metrics, compute_set_prf, parse_target_to_quads
from model import ByT5ACOSModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ByT5 ACOS Quadruple Extraction model.")
    parser.add_argument("--config", default="configs/stage_byt5_acos.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint best_model.pt")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="test", help="Split to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam size for generation (1 = greedy)")
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
    max_target_length: int = 384,
    num_beams: int = 1,
) -> dict:
    """
    Evaluates ByT5 model over a DataLoader and computes full ACOS metrics.
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

    sample_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating ACOS quads"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length,
                num_beams=num_beams,
            )

            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for pred_str, gold_quads, raw_text, gold_target_str in zip(
                pred_texts, batch["gold_quads"], batch["raw_texts"], batch["raw_targets"]
            ):
                pred_quads = parse_target_to_quads(pred_str)

                # 1. Full Quadruple (a, c, s, o)
                tp, fp, fn = compute_set_prf(pred_quads, gold_quads)
                counts["full_quad"]["tp"] += tp
                counts["full_quad"]["fp"] += fp
                counts["full_quad"]["fn"] += fn

                # 2. Explicit-only vs Implicit-only Quads
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

                # 3. Sub-tasks
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

                if len(sample_predictions) < 5:
                    sample_predictions.append({
                        "text": raw_text,
                        "pred_string": pred_str,
                        "gold_string": gold_target_str,
                        "pred_quads": pred_quads,
                        "gold_quads": gold_quads,
                    })

    metrics = {
        task: calculate_metrics(v["tp"], v["fp"], v["fn"])
        for task, v in counts.items()
    }
    metrics["sample_predictions"] = sample_predictions
    return metrics


def print_metrics_table(metrics: dict, title: str = "ByT5 ACOS Evaluation Results"):
    print("\n" + "=" * 80)
    print(f"  {title}")
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
        if task_key in metrics:
            m = metrics[task_key]
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
    print(f"Using device: {device}")

    model_cfg = cfg.get("model", {})
    max_source_length = model_cfg.get("max_source_length", 768)
    max_target_length = model_cfg.get("max_target_length", 384)

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
            num_beams=args.num_beams,
        )

        print_metrics_table(metrics, title=f"ByT5 ACOS Evaluation ({split.upper()} Split)")

        out_path = args.output_file or os.path.join(output_dir, f"{split}_metrics.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to: {out_path}")


if __name__ == "__main__":
    main()
