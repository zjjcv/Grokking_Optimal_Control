#!/usr/bin/env python3
"""
Compute Local Learning Coefficient (LLC) for saved checkpoints.

Single-seed legacy layout:
    data/{op}/checkpoints/checkpoint_step_{step}.pt
    data/{op}/llc.csv

Multi-seed layouts supported:
    data/{op}/seed_{seed}/checkpoints/checkpoint_step_{step}.pt
    data/seed_{seed}/{op}/checkpoints/checkpoint_step_{step}.pt

In multi-seed mode the script writes per-seed llc.csv files and an aggregate
data/{op}/llc_multiseed.csv.
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, GrokkingTransformer

from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary


OP_DIRS = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}

DEFAULT_SEEDS = [0, 1, 2]


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


def make_loader(p, operation, project_root, run_dir=None, batch_size=512, seed=42):
    legacy_path = os.path.join(project_root, "data", OP_DIRS[operation], "train_data.json")
    seed_path = os.path.join(run_dir, "train_data.json") if run_dir is not None else None
    data_path = seed_path if seed_path is not None and os.path.exists(seed_path) else legacy_path
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


def checkpoint_step(filename):
    return int(filename.split("_")[-1].split(".")[0])


def resolve_seed_run_dir(project_root, op_dir, seed, multi_seed):
    data_root = os.path.join(project_root, "data")
    legacy_dir = os.path.join(data_root, op_dir)
    candidates = []
    if multi_seed:
        candidates.extend([
            os.path.join(legacy_dir, f"seed_{seed}"),
            os.path.join(data_root, f"seed_{seed}", op_dir),
        ])
    candidates.append(legacy_dir)

    for run_dir in candidates:
        if os.path.isdir(os.path.join(run_dir, "checkpoints")):
            if multi_seed and run_dir == legacy_dir:
                print(f"[WARN] seed={seed}: using legacy checkpoints without a seed directory: {run_dir}")
            return run_dir
    return candidates[0]


def compute_seed_llc(args, config, project_root, op_dir, seed, device, aggregate_writer=None):
    set_seed(seed)
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    output_csv = os.path.join(run_dir, "llc.csv")

    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return

    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_step_") and f.endswith(".pt")],
        key=checkpoint_step,
    )
    if args.step_interval > 0:
        ckpt_files = [f for f in ckpt_files if checkpoint_step(f) % args.step_interval == 0]

    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")
    print(f"seed={seed}  device={device}  chains={args.num_chains}  draws={args.num_draws}")

    loader = make_loader(config.p, args.operation, project_root, run_dir=run_dir, seed=seed)

    os.makedirs(run_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "llc_mean", "llc_std", "seed"])
        f.flush()

        for idx, ckpt_file in enumerate(ckpt_files):
            step = checkpoint_step(ckpt_file)
            path = os.path.join(ckpt_dir, ckpt_file)

            state_dict = torch.load(path, map_location="cpu")["model_state_dict"]
            model = GrokkingTransformer(config).to(device)
            model.load_state_dict(state_dict)

            results = estimate_learning_coeff_with_summary(
                model=model,
                loader=loader,
                evaluate=evaluate,
                sampling_method=SGLD,
                optimizer_kwargs={"lr": args.lr, "noise_level": args.noise_level},
                num_draws=args.num_draws,
                num_chains=args.num_chains,
                num_burnin_steps=args.num_burnin_steps,
                device=device,
                verbose=False,
            )

            llc_mean = results["llc/mean"]
            llc_std = results["llc/std"]
            writer.writerow([step, f"{llc_mean:.6f}", f"{llc_std:.6f}", seed])
            if aggregate_writer is not None:
                aggregate_writer.writerow([seed, step, f"{llc_mean:.6f}", f"{llc_std:.6f}", run_dir])
            f.flush()

            print(
                f"  [{idx + 1}/{len(ckpt_files)}] seed={seed} step={step:6d} | "
                f"LLC = {llc_mean:.4f} +/- {llc_std:.4f}"
            )

    print(f"LLC data saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Compute LLC for grokking checkpoints")
    parser.add_argument("--operation", type=str, default="add", choices=list(OP_DIRS))
    parser.add_argument("--num-chains", type=int, default=5)
    parser.add_argument("--num-draws", type=int, default=100)
    parser.add_argument("--num-burnin-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--step-interval", type=int, default=0,
                        help="Only process checkpoints whose step is a multiple of this value (0=all).")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run LLC for multiple seed directories. Default seeds: 0 1 2.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Single-seed mode SGLD/data-loader seed.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seeds used when --multi-seed is enabled.")
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIRS[args.operation]
    seeds = args.seeds if args.multi_seed else [args.seed]

    aggregate_file = None
    aggregate_writer = None
    aggregate_path = os.path.join(project_root, "data", op_dir, "llc_multiseed.csv")
    if args.multi_seed:
        os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)
        aggregate_file = open(aggregate_path, "w", newline="")
        aggregate_writer = csv.writer(aggregate_file)
        aggregate_writer.writerow(["seed", "step", "llc_mean", "llc_std", "run_dir"])

    try:
        for seed in seeds:
            compute_seed_llc(args, config, project_root, op_dir, seed, device, aggregate_writer)
    finally:
        if aggregate_file is not None:
            aggregate_file.close()
            print(f"\nMulti-seed LLC data saved to: {aggregate_path}")


if __name__ == "__main__":
    main()
