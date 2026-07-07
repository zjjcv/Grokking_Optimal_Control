#!/usr/bin/env python3
"""
LLC robustness sweep over SGLD hyperparameters.

Grid:
    lr = 5e-6, 1e-5, 2e-5
    noise_level = 0.5, 1.0, 2.0
    seed = 0, 1, 2
    steps = 100, 500, 1000, 3000, 10000, 30000, 50000, 90000

Outputs:
    data/{op}/llc_robustness/llc_robustness_raw.csv
    data/{op}/llc_robustness/llc_robustness_summary.csv

Usage:
    python src/llc_robustness.py --operation add
    python src/llc_robustness.py --operation all
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, GrokkingTransformer

from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary


OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}

DEFAULT_LRS = [5e-6, 1e-5, 2e-5]
DEFAULT_NOISE_LEVELS = [0.5, 1.0, 2.0]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_STEPS = [100, 500, 1000, 3000, 10000, 30000, 50000, 90000]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data):
    inputs, labels = data
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)
    return loss, {}


def make_loader(p, operation, project_root, batch_size=512, seed=42):
    op_dir = OP_DIR[operation]
    data_path = os.path.join(project_root, "data", op_dir, "train_data.json")
    with open(data_path, "r") as f:
        pairs = json.load(f)

    xs, ys, labels = [], [], []
    for x, y in pairs:
        xs.append(x)
        ys.append(y)
        if operation == "add":
            labels.append((x + y) % p)
        elif operation == "sub":
            labels.append((x - y) % p)
        elif operation == "mul":
            labels.append((x * y) % p)
        elif operation == "div":
            labels.append(0 if y == 0 else (x * pow(y, -1, p)) % p)

    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(inputs, labels_t)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def load_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["step"])].append(float(row["llc_mean"]))

    summary = []
    for step in sorted(groups):
        vals = np.array(groups[step], dtype=np.float64)
        summary.append({
            "step": step,
            "llc_mean": float(vals.mean()),
            "llc_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "llc_min": float(vals.min()),
            "llc_max": float(vals.max()),
            "n": int(len(vals)),
        })
    return summary


def process_operation(args, operation):
    config = Config(operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[operation]
    ckpt_dir = os.path.join(project_root, "data", op_dir, "checkpoints")
    out_dir = os.path.join(project_root, "data", op_dir, "llc_robustness")
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "llc_robustness_raw.csv")
    summary_path = os.path.join(out_dir, "llc_robustness_summary.csv")

    raw_fields = [
        "operation",
        "step",
        "lr",
        "noise_level",
        "seed",
        "llc_mean",
        "llc_std",
        "num_chains",
        "num_draws",
        "num_burnin_steps",
    ]

    rows = []
    total = len(args.steps) * len(args.lrs) * len(args.noise_levels) * len(args.seeds)
    done = 0

    print("=" * 60)
    print("LLC robustness sweep")
    print(f"operation  = {operation}")
    print(f"device     = {device}")
    print(f"steps      = {args.steps}")
    print(f"lrs        = {args.lrs}")
    print(f"noise      = {args.noise_levels}")
    print(f"seeds      = {args.seeds}")
    print(f"output     = {raw_path}")
    print("=" * 60)

    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        f.flush()

        for step in args.steps:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] step {step}: checkpoint not found")
                continue

            state_dict = load_state_dict(ckpt_path)

            for lr in args.lrs:
                for noise_level in args.noise_levels:
                    for seed in args.seeds:
                        done += 1
                        set_seed(seed)
                        loader = make_loader(
                            config.p,
                            operation,
                            project_root,
                            batch_size=args.batch_size,
                            seed=seed,
                        )
                        model = GrokkingTransformer(config).to(device)
                        model.load_state_dict(state_dict)

                        results = estimate_learning_coeff_with_summary(
                            model=model,
                            loader=loader,
                            evaluate=evaluate,
                            sampling_method=SGLD,
                            optimizer_kwargs={
                                "lr": lr,
                                "noise_level": noise_level,
                            },
                            num_draws=args.num_draws,
                            num_chains=args.num_chains,
                            num_burnin_steps=args.num_burnin_steps,
                            device=device,
                            verbose=False,
                        )

                        row = {
                            "operation": operation,
                            "step": step,
                            "lr": f"{lr:.8g}",
                            "noise_level": f"{noise_level:.8g}",
                            "seed": seed,
                            "llc_mean": f"{results['llc/mean']:.8f}",
                            "llc_std": f"{results['llc/std']:.8f}",
                            "num_chains": args.num_chains,
                            "num_draws": args.num_draws,
                            "num_burnin_steps": args.num_burnin_steps,
                        }
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()

                        print(
                            f"[{done}/{total}] step={step:6d} lr={lr:.1e} "
                            f"noise={noise_level:.1f} seed={seed} "
                            f"LLC={results['llc/mean']:.4f} +/- {results['llc/std']:.4f}"
                        )

    summary = summarize(rows)
    with open(summary_path, "w", newline="") as f:
        fields = ["step", "llc_mean", "llc_std", "llc_min", "llc_max", "n"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({
                "step": row["step"],
                "llc_mean": f"{row['llc_mean']:.8f}",
                "llc_std": f"{row['llc_std']:.8f}",
                "llc_min": f"{row['llc_min']:.8f}",
                "llc_max": f"{row['llc_max']:.8f}",
                "n": row["n"],
            })

    print(f"Saved raw: {raw_path}")
    print(f"Saved summary: {summary_path}\n")


def main():
    parser = argparse.ArgumentParser(description="LLC robustness sweep")
    parser.add_argument("--operation", choices=list(OP_DIR.keys()) + ["all"], default="add")
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--lrs", type=float, nargs="+", default=list(DEFAULT_LRS))
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-chains", type=int, default=5)
    parser.add_argument("--num-draws", type=int, default=100)
    parser.add_argument("--num-burnin-steps", type=int, default=20)
    args = parser.parse_args()

    operations = list(OP_DIR.keys()) if args.operation == "all" else [args.operation]
    for operation in operations:
        process_operation(args, operation)


if __name__ == "__main__":
    main()
