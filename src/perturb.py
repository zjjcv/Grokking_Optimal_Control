#!/usr/bin/env python3
"""
Perturbation and random-ablation robustness for Transformer grokking.

Single-seed legacy layout:
    data/{op}/checkpoints/
    data/{op}/train_data.json
    data/{op}/test_data.json

Multi-seed layouts supported:
    data/{op}/seed_{seed}/checkpoints/
    data/seed_{seed}/{op}/checkpoints/

For each checkpoint, computes original train/test accuracy, Gaussian weight
perturbation accuracy, and a random weight-ablation control.
"""

import argparse
import csv
import json
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train import Config, GrokkingTransformer


OP_DIR = {
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
                print(f"[WARN] seed={seed}: using legacy checkpoints without seed directory: {run_dir}")
            return run_dir
    return candidates[0]


def make_loader(p, operation, split, project_root, run_dir, batch_size=512):
    op_dir = OP_DIR[operation]
    seed_path = os.path.join(run_dir, f"{split}_data.json")
    legacy_path = os.path.join(project_root, "data", op_dir, f"{split}_data.json")
    data_path = seed_path if os.path.exists(seed_path) else legacy_path
    with open(data_path, "r") as f:
        pairs = json.load(f)

    xs, ys, labels = [], [], []
    for x, y in pairs:
        if operation == "div" and y == 0:
            continue
        xs.append(x)
        ys.append(y)
        if operation == "add":
            labels.append((x + y) % p)
        elif operation == "sub":
            labels.append((x - y) % p)
        elif operation == "mul":
            labels.append((x * y) % p)
        elif operation == "div":
            labels.append((x * pow(y, -1, p)) % p)

    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, labels_t), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def is_float_tensor(tensor):
    return torch.is_tensor(tensor) and torch.is_floating_point(tensor)


def perturb_state_dict(sd, epsilon):
    new_sd = {}
    for key, tensor in sd.items():
        if not is_float_tensor(tensor):
            new_sd[key] = tensor.clone()
            continue
        std = tensor.float().std().item()
        if std == 0:
            new_sd[key] = tensor.clone()
            continue
        new_sd[key] = tensor + torch.randn_like(tensor) * epsilon * std
    return new_sd


def random_ablate_state_dict(sd, fraction):
    fraction = min(max(fraction, 0.0), 1.0)
    new_sd = {}
    for key, tensor in sd.items():
        if not is_float_tensor(tensor):
            new_sd[key] = tensor.clone()
            continue
        mask = torch.rand_like(tensor.float()) >= fraction
        new_sd[key] = tensor * mask.to(dtype=tensor.dtype)
    return new_sd


def discover_steps(ckpt_dir):
    steps = []
    for filename in os.listdir(ckpt_dir):
        match = re.match(r"checkpoint_step_(\d+)\.pt", filename)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def format_float(value):
    return "" if np.isnan(value) else f"{value:.6f}"


FIELDS = [
    "seed",
    "step",
    "train_acc_orig",
    "test_acc_orig",
    "train_acc_perturb",
    "test_acc_perturb",
    "train_acc_perturb_std",
    "test_acc_perturb_std",
    "train_acc_random_ablation",
    "test_acc_random_ablation",
    "train_acc_random_ablation_std",
    "test_acc_random_ablation_std",
    "train_drop_perturb",
    "test_drop_perturb",
    "train_drop_random_ablation",
    "test_drop_random_ablation",
    "epsilon",
    "ablation_fraction",
    "perturb_trials",
    "random_ablation_trials",
]

TRIAL_FIELDS = [
    "seed",
    "step",
    "trial",
    "train_acc_orig",
    "test_acc_orig",
    "train_acc_random_ablation",
    "test_acc_random_ablation",
    "train_drop_random_ablation",
    "test_drop_random_ablation",
    "ablation_fraction",
]


def process_seed(args, config, project_root, op_dir, seed, device):
    set_seed(seed)
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return [], []

    out_dir = os.path.join(run_dir, "perturb")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "perturb.csv")
    trials_csv = os.path.join(out_dir, "random_ablation_trials.csv")
    steps = args.steps or discover_steps(ckpt_dir)

    train_loader = make_loader(config.p, args.operation, "train", project_root, run_dir, args.batch_size)
    test_loader = make_loader(config.p, args.operation, "test", project_root, run_dir, args.batch_size)
    model = GrokkingTransformer(config).to(device)

    print(
        f"seed={seed} run_dir={run_dir} epsilon={args.epsilon} perturb_trials={args.trials} "
        f"ablation_fraction={args.ablation_fraction} random_ablation_trials={args.random_ablation_trials} "
        f"steps={len(steps)} device={device}"
    )

    summary_rows = []
    trial_rows = []
    with open(out_csv, "w", newline="") as f_sum, open(trials_csv, "w", newline="") as f_trials:
        summary_writer = csv.DictWriter(f_sum, fieldnames=FIELDS)
        trial_writer = csv.DictWriter(f_trials, fieldnames=TRIAL_FIELDS)
        summary_writer.writeheader()
        trial_writer.writeheader()

        for idx, step in enumerate(steps):
            path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            if not os.path.exists(path):
                continue
            state = torch.load(path, map_location="cpu")
            sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

            model.load_state_dict(sd)
            train_orig = evaluate(model, train_loader, device)
            test_orig = evaluate(model, test_loader, device)

            train_perturbs, test_perturbs = [], []
            for _ in range(args.trials):
                model.load_state_dict(perturb_state_dict(sd, args.epsilon))
                train_perturbs.append(evaluate(model, train_loader, device))
                test_perturbs.append(evaluate(model, test_loader, device))
            train_perturb, train_perturb_std = mean_std(train_perturbs)
            test_perturb, test_perturb_std = mean_std(test_perturbs)

            train_randoms, test_randoms = [], []
            for trial in range(args.random_ablation_trials):
                model.load_state_dict(random_ablate_state_dict(sd, args.ablation_fraction))
                train_random = evaluate(model, train_loader, device)
                test_random = evaluate(model, test_loader, device)
                train_randoms.append(train_random)
                test_randoms.append(test_random)
                trial_row = {
                    "seed": seed,
                    "step": step,
                    "trial": trial,
                    "train_acc_orig": f"{train_orig:.6f}",
                    "test_acc_orig": f"{test_orig:.6f}",
                    "train_acc_random_ablation": f"{train_random:.6f}",
                    "test_acc_random_ablation": f"{test_random:.6f}",
                    "train_drop_random_ablation": f"{train_orig - train_random:.6f}",
                    "test_drop_random_ablation": f"{test_orig - test_random:.6f}",
                    "ablation_fraction": f"{args.ablation_fraction:.6f}",
                }
                trial_writer.writerow(trial_row)
                trial_rows.append(trial_row)

            train_random, train_random_std = mean_std(train_randoms)
            test_random, test_random_std = mean_std(test_randoms)
            row = {
                "seed": seed,
                "step": step,
                "train_acc_orig": f"{train_orig:.6f}",
                "test_acc_orig": f"{test_orig:.6f}",
                "train_acc_perturb": format_float(train_perturb),
                "test_acc_perturb": format_float(test_perturb),
                "train_acc_perturb_std": format_float(train_perturb_std),
                "test_acc_perturb_std": format_float(test_perturb_std),
                "train_acc_random_ablation": format_float(train_random),
                "test_acc_random_ablation": format_float(test_random),
                "train_acc_random_ablation_std": format_float(train_random_std),
                "test_acc_random_ablation_std": format_float(test_random_std),
                "train_drop_perturb": format_float(train_orig - train_perturb),
                "test_drop_perturb": format_float(test_orig - test_perturb),
                "train_drop_random_ablation": format_float(train_orig - train_random),
                "test_drop_random_ablation": format_float(test_orig - test_random),
                "epsilon": f"{args.epsilon:.6f}",
                "ablation_fraction": f"{args.ablation_fraction:.6f}",
                "perturb_trials": args.trials,
                "random_ablation_trials": args.random_ablation_trials,
            }
            summary_writer.writerow(row)
            summary_rows.append(row)
            f_sum.flush()
            f_trials.flush()

            if (idx + 1) % 100 == 0 or idx == 0 or step == steps[-1]:
                print(
                    f"  [{idx + 1}/{len(steps)}] seed={seed} step={step:6d} "
                    f"orig test={test_orig:.4f} | noise test={test_perturb:.4f} | "
                    f"random ablation test={test_random:.4f} +/- {test_random_std:.4f}"
                )

    print(f"[OK] {out_csv}")
    print(f"[OK] {trials_csv}")
    return summary_rows, trial_rows


def write_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {path}")


def main():
    parser = argparse.ArgumentParser(description="Perturbation robustness analysis")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--ablation-fraction", type=float, default=0.01)
    parser.add_argument("--random-ablation-trials", type=int, default=20)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = args.seeds if args.multi_seed else [args.seed]

    all_summary, all_trials = [], []
    for seed in seeds:
        summary_rows, trial_rows = process_seed(args, config, project_root, op_dir, seed, device)
        all_summary.extend(summary_rows)
        all_trials.extend(trial_rows)

    if args.multi_seed and all_summary:
        out_dir = os.path.join(project_root, "data", op_dir, "perturb")
        write_rows(os.path.join(out_dir, "perturb_multiseed.csv"), all_summary, FIELDS)
        write_rows(os.path.join(out_dir, "random_ablation_trials_multiseed.csv"), all_trials, TRIAL_FIELDS)
    elif all_summary:
        legacy = os.path.join(project_root, "data", op_dir, "perturb.csv")
        write_rows(legacy, all_summary, FIELDS)


if __name__ == "__main__":
    main()
