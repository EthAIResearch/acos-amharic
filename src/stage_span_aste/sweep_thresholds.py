"""
Relation Probability Threshold Sweep for Span-ASTE (ACL 2021)
==============================================================
Evaluates an existing Span-ASTE checkpoint across a range of relation probability
thresholds tau in a single inference pass without re-running the Transformer encoder:
  P(Relation) = 1.0 - P(INVALID) >= tau

Usage:
    PYTHONPATH=src/stage_span_aste python src/stage_span_aste/sweep_thresholds.py \
        --config configs/stage_span_aste.yaml \
        --checkpoint results/stage_span_aste/afroxlmr_run1/best_model.pt \
        --split both
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
from dataset import SpanASTEDataset, collate_fn
from evaluation import sweep_relation_thresholds
from model import SpanASTEModel


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep relation thresholds on Span-ASTE checkpoint.")
    parser.add_argument("--config", default="configs/stage_span_aste.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint best_model.pt")
    parser.add_argument("--output_dir", default=None, help="Output directory to save sweep JSON results")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both", help="Which split(s) to evaluate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None, help="Custom list of thresholds")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


@torch.no_grad()
def collect_candidate_predictions(model, dataloader, device):
    """
    Runs model in evaluation mode over a dataset and extracts
    (candidate_pairs_with_scores, raw_quads) for all examples.
    """
    model.eval()
    all_candidates_with_scores = []
    all_raw_quads = []

    for batch in tqdm(dataloader, desc="Collecting candidate scores"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        subword_to_word = batch["subword_to_word"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            subword_to_word=subword_to_word,
            seq_spans=batch["seq_spans"],
            gold_mention_labels=None,
            gold_pairs=None,
            num_words=batch["num_words"],
            relation_threshold=None,
        )

        for b_out, quads in zip(out["batch_outputs"], batch["raw_quads"]):
            all_candidates_with_scores.append(b_out.get("candidate_pairs_with_scores", []))
            all_raw_quads.append(quads)

    return all_candidates_with_scores, all_raw_quads


def print_sweep_table(split_name: str, sweep_results: dict):
    print("\n=======================================================")
    print(f"  Relation Threshold Sweep ({split_name.upper()} Split)")
    print("=======================================================")
    print(f"  Total Gold Pairs        : {sweep_results['total_gold_pairs']}")
    print(f"  Reachable Candidate Pairs: {sweep_results['reachable_pairs']} "
          f"({sweep_results['candidate_ceiling_recall'] * 100:.2f}% Ceiling Recall)")
    print(f"  Best AOPE Threshold    : tau = {sweep_results['best_aope_threshold']} "
          f"(F1 = {sweep_results['best_aope_f1'] * 100:.2f}%)")
    print(f"  Best ASTE Threshold    : tau = {sweep_results['best_aste_threshold']} "
          f"(F1 = {sweep_results['best_aste_f1'] * 100:.2f}%)")
    print("------------------------------------------------------------------------------------------------")
    print(f"  {'tau':<6} | {'AOPE Prec':<10} | {'AOPE Rec':<9} | {'AOPE F1':<9} | "
          f"{'ASTE Prec':<10} | {'ASTE Rec':<9} | {'ASTE F1':<9} | {'Conversion':<10}")
    print("------------------------------------------------------------------------------------------------")

    for row in sweep_results["threshold_sweep"]:
        t = row["threshold"]
        a_p = row["aope_precision"] * 100
        a_r = row["aope_recall"] * 100
        a_f1 = row["aope_f1"] * 100
        s_p = row["aste_precision"] * 100
        s_r = row["aste_recall"] * 100
        s_f1 = row["aste_f1"] * 100
        conv = row["conversion_efficiency"] * 100

        marker = ""
        if t == sweep_results["best_aope_threshold"]:
            marker = " ★ (Best AOPE)"
        elif t == sweep_results["best_aste_threshold"]:
            marker = " ◆ (Best ASTE)"

        print(f"  {t:<6.2f} | {a_p:<9.2f}% | {a_r:<8.2f}% | {a_f1:<8.2f}% | "
              f"{s_p:<9.2f}% | {s_r:<8.2f}% | {s_f1:<8.2f}% | {conv:<9.2f}%{marker}")

    print("------------------------------------------------------------------------------------------------")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg.get("model_name", "Davlan/afro-xlmr-base")
    default_dir = cfg.get("output_dir", "results/stage_span_aste/afroxlmr_run1")
    checkpoint_path = args.checkpoint or os.path.join(default_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Determine output directory (priority: CLI arg > checkpoint parent directory > config output_dir)
    output_dir = args.output_dir
    if not output_dir:
        if args.checkpoint:
            output_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        else:
            output_dir = default_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for sweep results: {output_dir}")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Model & Data Hyperparameters
    model_cfg = cfg.get("model", {})
    max_length = model_cfg.get("max_length", 256)
    max_words = model_cfg.get("max_words", 128)
    max_span_length = model_cfg.get("max_span_length", 8)
    pruning_ratio = model_cfg.get("pruning_ratio", 0.5)
    width_dim = model_cfg.get("width_dim", 20)
    distance_dim = model_cfg.get("distance_dim", 128)
    hidden_dim = model_cfg.get("hidden_dim", 150)
    dropout = model_cfg.get("dropout", 0.4)

    training_cfg = cfg.get("training", {})
    batch_size = args.batch_size or training_cfg.get("batch_size", 4)

    data_cfg = cfg.get("data", {})
    dev_path = data_cfg.get("dev", "data/prepared_explicit/dev.jsonl")
    test_path = data_cfg.get("test", "data/prepared_explicit/test.jsonl")

    use_biaffine = model_cfg.get("use_biaffine", True)
    biaffine_dim = model_cfg.get("biaffine_dim", 256)
    use_span_sync = model_cfg.get("use_span_sync", True)

    print(f"Loading checkpoint weights from {checkpoint_path}...")
    loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
    else:
        state_dict = loaded

    # Auto-detect whether checkpoint was trained with Biaffine or legacy MLP
    if any(k.startswith("relation_classifier.net.") for k in state_dict):
        print("  -> Detected legacy MLP relation classifier checkpoint (use_biaffine=False).")
        use_biaffine = False
    elif any(k.startswith("relation_classifier.U") for k in state_dict):
        print("  -> Detected Deep Biaffine relation classifier checkpoint (use_biaffine=True).")
        use_biaffine = True
        u_tensor = state_dict.get("relation_classifier.U")
        if u_tensor is not None:
            biaffine_dim = u_tensor.shape[1]
        affine_weight = state_dict.get("relation_classifier.affine_classifier.0.weight")
        if affine_weight is not None:
            use_span_sync = affine_weight.shape[1] == (biaffine_dim * 5 + distance_dim)
        print(f"     Architecture configuration: biaffine_dim={biaffine_dim}, span_sync={use_span_sync}")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    )
    model.load_state_dict(state_dict)
    model.to(device)

    splits_to_run = ["dev", "test"] if args.split == "both" else [args.split]
    best_dev_threshold = 0.5

    # 1. Dev Sweep
    if "dev" in splits_to_run:
        print("\nLoading Dev dataset...")
        dev_dataset = SpanASTEDataset(
            dev_path,
            tokenizer,
            max_length=max_length,
            max_words=max_words,
            max_span_length=max_span_length,
        )
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        dev_cands, dev_quads = collect_candidate_predictions(model, dev_loader, device)
        dev_sweep = sweep_relation_thresholds(dev_cands, dev_quads, thresholds=args.thresholds)
        print_sweep_table("dev", dev_sweep)

        best_dev_threshold = dev_sweep["best_aope_threshold"]
        dev_out_path = os.path.join(output_dir, "dev_threshold_sweep.json")
        with open(dev_out_path, "w", encoding="utf-8") as f:
            json.dump(dev_sweep, f, indent=2)
        print(f"Saved Dev sweep to: {dev_out_path}")

    # 2. Test Sweep
    if "test" in splits_to_run:
        print("\nLoading Test dataset...")
        test_dataset = SpanASTEDataset(
            test_path,
            tokenizer,
            max_length=max_length,
            max_words=max_words,
            max_span_length=max_span_length,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_cands, test_quads = collect_candidate_predictions(model, test_loader, device)
        test_sweep = sweep_relation_thresholds(test_cands, test_quads, thresholds=args.thresholds)
        print_sweep_table("test", test_sweep)

        # Highlight performance at dev-tuned optimal threshold (strict ML protocol)
        tuned_test_metric = next(
            (r for r in test_sweep["threshold_sweep"] if r["threshold"] == best_dev_threshold),
            None,
        )
        if tuned_test_metric:
            test_sweep["dev_tuned_threshold"] = best_dev_threshold
            test_sweep["dev_tuned_test_metrics"] = tuned_test_metric
            print(f"\n>>> Test performance at Dev-Tuned Threshold (tau = {best_dev_threshold}):")
            print(f"    AOPE F1: {tuned_test_metric['aope_f1'] * 100:.2f}% "
                  f"(P={tuned_test_metric['aope_precision'] * 100:.2f}%, R={tuned_test_metric['aope_recall'] * 100:.2f}%)")
            print(f"    ASTE F1: {tuned_test_metric['aste_f1'] * 100:.2f}% "
                  f"(P={tuned_test_metric['aste_precision'] * 100:.2f}%, R={tuned_test_metric['aste_recall'] * 100:.2f}%)")
            print(f"    Conversion: {tuned_test_metric['conversion_efficiency'] * 100:.2f}%")

        test_out_path = os.path.join(output_dir, "threshold_sweep.json")
        with open(test_out_path, "w", encoding="utf-8") as f:
            json.dump(test_sweep, f, indent=2)
        print(f"Saved Test sweep to: {test_out_path}")


if __name__ == "__main__":
    main()
