#!/usr/bin/env python3
"""
Run signature analyses on hyperparameter robustness sweep runs.

For each config under data/{op}/robustness/{cfg}/seed_{s}/:
  - relation (OV-NFM structure vs CMA, sparse steps)
  - Hessian probes (sparse steps)
  - spectral entropy (sparse steps)

Also writes an aggregate summary CSV at
    data/{op}/robustness/signatures_summary.csv

Usage:
    python src/robustness_signatures.py
    python src/robustness_signatures.py --config wd_0.001 --seed 0
    python src/robustness_signatures.py --steps 5000 30000 90000
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness_common import (
    DEFAULT_SEEDS,
    OP_DIR,
    SIGNATURE_STEPS,
    detect_grokking_step,
    discover_robustness_runs,
    final_test_acc,
    load_config_from_run_dir,
    load_metric_csv,
)
from relation import process_step as relation_process_step
from hessian_probe import (
    TOP_K,
    checkpoint_step,
    load_train_tensors,
    probe_checkpoint,
)
from spectral_entropy import discover_steps, load_state_dict, spectral_entropy


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[OK] {path}")


def run_relation(args, config, project_root, op_dir, config_name, seed, run_dir, steps, device):
    relation_dir = os.path.join(run_dir, "relation")
    all_head_rows = []
    step_rows = []
    for step in steps:
        head_rows, step_row = relation_process_step(
            args, config, project_root, op_dir, run_dir, seed, step, device
        )
        if head_rows is None:
            continue
        for row in head_rows:
            row["config"] = config_name
        step_row["config"] = config_name
        all_head_rows.extend(head_rows)
        step_rows.append(step_row)

    if all_head_rows:
        head_fields = list(all_head_rows[0].keys())
        step_fields = list(step_rows[0].keys())
        write_csv(os.path.join(relation_dir, "relation_heads.csv"), all_head_rows, head_fields)
        write_csv(os.path.join(relation_dir, "relation.csv"), step_rows, step_fields)
    return all_head_rows, step_rows


def run_hessian(args, config, project_root, operation, config_name, seed, run_dir, steps, device):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_csv = os.path.join(run_dir, "hessian.csv")
    if not os.path.isdir(ckpt_dir):
        return []

    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_step_") and f.endswith(".pt")],
        key=checkpoint_step,
    )
    wanted = set(steps)
    ckpt_files = [f for f in ckpt_files if checkpoint_step(f) in wanted]
    if not ckpt_files:
        return []

    inputs, labels = load_train_tensors(config.p, operation, project_root, run_dir=run_dir)
    inputs, labels = inputs.to(device), labels.to(device)

    base_cols = ["step", "seed", "config", "loss", "lambda_max", "lambda_min",
                 "trace", "trace_sq", "erank_pr", "erank_max"]
    eig_cols = [f"lambda_top_{i + 1}" for i in range(TOP_K)]
    rows = []

    for ckpt_file in ckpt_files:
        step = checkpoint_step(ckpt_file)
        path = os.path.join(ckpt_dir, ckpt_file)
        res = probe_checkpoint(
            config, path, inputs, labels, device, args,
            probe_seed=seed * 1000003 + step,
        )
        row = {
            "step": step,
            "seed": seed,
            "config": config_name,
            "loss": res["loss"],
            "lambda_max": res["lambda_max"],
            "lambda_min": res["lambda_min"],
            "trace": res["trace"],
            "trace_sq": res["trace_sq"],
            "erank_pr": res["erank_pr"],
            "erank_max": res["erank_max"],
        }
        for i, val in enumerate(res["top_eigs"]):
            row[f"lambda_top_{i + 1}"] = val
        rows.append(row)
        print(
            f"  hessian cfg={config_name} seed={seed} step={step} "
            f"erank_pr={res['erank_pr']:.2f} trace={res['trace']:.4f}"
        )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + eig_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"[OK] {out_csv}")
    return rows


def run_spectral_entropy(operation, config_name, seed, run_dir, steps, device):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_csv = os.path.join(run_dir, "spectral_entropy.csv")
    if not os.path.isdir(ckpt_dir):
        return

    available = discover_steps(ckpt_dir)
    use_steps = [s for s in steps if s in available]
    if not use_steps:
        return

    param_names = None
    for step in use_steps:
        path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
        if os.path.exists(path):
            param_names = sorted(load_state_dict(path).keys())
            break
    if param_names is None:
        return

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "seed", "config", "mean_entropy"] + param_names)
        for step in use_steps:
            path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            if not os.path.exists(path):
                continue
            state_dict = load_state_dict(path)
            entropies = [spectral_entropy(state_dict[name], device) for name in param_names]
            writer.writerow(
                [step, seed, config_name, f"{sum(entropies) / len(entropies):.6f}"]
                + [f"{e:.6f}" for e in entropies]
            )
    print(f"[OK] {out_csv}")


def process_run(args, project_root, operation, config_name, seed, run_dir, steps, device):
    config = load_config_from_run_dir(run_dir, operation)
    op_dir = OP_DIR[operation]
    print(f"\n--- {config_name} seed={seed} ---")

    metric_rows = load_metric_csv(config.metric_file)
    grok_step = detect_grokking_step(metric_rows)
    fin_acc = final_test_acc(metric_rows)

    _, step_rows = run_relation(
        args, config, project_root, op_dir, config_name, seed, run_dir, steps, device
    )
    hessian_rows = run_hessian(
        args, config, project_root, operation, config_name, seed, run_dir, steps, device
    )
    run_spectral_entropy(operation, config_name, seed, run_dir, steps, device)

    # peak erank at grokking-adjacent step if available
    peak_erank = None
    if hessian_rows:
        peak_erank = max(r["erank_pr"] for r in hessian_rows)

    late_pearson = None
    late_spearman = None
    if step_rows:
        late = max(step_rows, key=lambda r: r["step"])
        late_pearson = late.get("pearson")
        late_spearman = late.get("spearman")

    return {
        "config": config_name,
        "seed": seed,
        "run_dir": run_dir,
        "grokking_step": grok_step if grok_step is not None else "",
        "final_test_acc": fin_acc if fin_acc is not None else "",
        "peak_erank_pr": peak_erank if peak_erank is not None else "",
        "late_pearson": late_pearson if late_pearson is not None else "",
        "late_spearman": late_spearman if late_spearman is not None else "",
        "num_layers": config.num_layers,
        "attention_dim": config.attention_dim,
        "weight_decay": config.weight_decay,
        "init_scale": config.init_scale,
        "train_ratio": config.train_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Signature analyses for robustness sweep")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--config", type=str, default=None, help="Single config name.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, nargs="+", default=SIGNATURE_STEPS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--lanczos-steps", type=int, default=64)
    parser.add_argument("--hutchinson-probes", type=int, default=32)
    parser.add_argument("--all", action="store_true", help="Process all discovered runs.")
    args = parser.parse_args()
    args.operation = args.operation  # relation.process_step reads this

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    op_dir = OP_DIR[args.operation]

    if args.config is not None:
        seeds = [args.seed] if args.seed is not None else args.seeds
        runs = [
            (args.config, s, os.path.join(project_root, "data", op_dir, "robustness", args.config, f"seed_{s}"))
            for s in seeds
        ]
    elif args.all:
        runs = discover_robustness_runs(project_root, args.operation, seeds=args.seeds)
    else:
        runs = discover_robustness_runs(project_root, args.operation, seeds=args.seeds)
        if not runs:
            print("[INFO] No runs found yet. Launch training with run_robustness_sweep.sh first.")
            return

    summary_rows = []
    for config_name, seed, run_dir in runs:
        if not os.path.isdir(os.path.join(run_dir, "checkpoints")):
            print(f"[SKIP] {config_name} seed={seed}: no checkpoints")
            continue
        summary_rows.append(
            process_run(args, project_root, args.operation, config_name, seed, run_dir, args.steps, device)
        )

    if summary_rows:
        summary_path = os.path.join(project_root, "data", op_dir, "robustness", "signatures_summary.csv")
        write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))


if __name__ == "__main__":
    main()
