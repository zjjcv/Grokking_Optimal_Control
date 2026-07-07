#!/usr/bin/env python3
"""
Linear and curved interpolation between memorization and generalization checkpoints.

Single-seed legacy layout:
    data/{op}/checkpoints/

Multi-seed layouts supported:
    data/{op}/seed_{seed}/checkpoints/
    data/seed_{seed}/{op}/checkpoints/

Each seed writes its own interpolation CSVs. In multi-seed mode, aggregate
copies are also written under data/{op}/interpolation/.
"""

import argparse
import csv
import json
import os
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train import Config, GrokkingTransformer


OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}
DEFAULT_SEEDS = [0, 1, 2]


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
    if split in ("train", "test"):
        seed_path = os.path.join(run_dir, f"{split}_data.json")
        legacy_path = os.path.join(project_root, "data", op_dir, f"{split}_data.json")
        data_path = seed_path if os.path.exists(seed_path) else legacy_path
        with open(data_path, "r") as f:
            pairs = json.load(f)
    else:
        pairs = [(x, y) for x in range(p) for y in range(p)
                 if not (operation == "div" and y == 0)]

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
    total_loss = 0.0
    correct = 0
    margin_sum = 0.0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()

        correct_logits = logits[torch.arange(logits.size(0)), labels]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(logits.size(0)), labels] = False
        wrong_max = logits.masked_fill(~mask, float("-inf")).max(dim=-1).values
        margin = correct_logits - wrong_max

        total_loss += loss.item()
        margin_sum += margin.sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total, margin_sum / total


def discover_steps(ckpt_dir):
    steps = []
    for filename in os.listdir(ckpt_dir):
        match = re.match(r"checkpoint_step_(\d+)\.pt", filename)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def nearest_step(steps, target):
    return min(steps, key=lambda s: abs(s - target))


def resolve_step(steps, requested, label):
    if requested in steps:
        return requested
    resolved = nearest_step(steps, requested)
    print(f"[WARN] {label} step {requested} not found; using nearest checkpoint {resolved}")
    return resolved


def load_sd(ckpt_dir, step):
    path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
    state = torch.load(path, map_location="cpu")
    return state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state


def linear_state(sd_start, sd_end, alpha):
    return {key: (1 - alpha) * sd_start[key] + alpha * sd_end[key] for key in sd_start}


def bezier_state(sd_start, sd_mid, sd_end, alpha):
    a = float(alpha)
    return {
        key: ((1 - a) ** 2) * sd_start[key] + 2 * (1 - a) * a * sd_mid[key] + (a ** 2) * sd_end[key]
        for key in sd_start
    }


def path_state(path_name, sd_start, sd_mid, sd_end, alpha):
    if path_name == "linear":
        return linear_state(sd_start, sd_end, alpha)
    if path_name == "bezier":
        return bezier_state(sd_start, sd_mid, sd_end, alpha)
    raise ValueError(f"Unknown path: {path_name}")


def summarize_path(rows, path_name, seed):
    path_rows = [r for r in rows if r["path"] == path_name]
    if not path_rows:
        return None
    start = min(path_rows, key=lambda r: r["alpha"])
    end = max(path_rows, key=lambda r: r["alpha"])
    min_endpoint_acc = min(start["test_acc"], end["test_acc"])
    max_endpoint_loss = max(start["test_loss"], end["test_loss"])
    min_endpoint_margin = min(start["test_margin"], end["test_margin"])
    min_test_acc = min(r["test_acc"] for r in path_rows)
    max_test_loss = max(r["test_loss"] for r in path_rows)
    min_test_margin = min(r["test_margin"] for r in path_rows)
    return {
        "seed": seed,
        "path": path_name,
        "min_test_acc": min_test_acc,
        "test_acc_barrier": min_endpoint_acc - min_test_acc,
        "max_test_loss": max_test_loss,
        "test_loss_barrier": max_test_loss - max_endpoint_loss,
        "min_test_margin": min_test_margin,
        "test_margin_barrier": min_endpoint_margin - min_test_margin,
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                formatted[key] = f"{value:.6f}" if isinstance(value, float) else value
            writer.writerow(formatted)
    print(f"[OK] {path}")


PATH_FIELDS = [
    "seed", "path", "alpha", "start_step", "mid_step", "end_step",
    "train_loss", "test_loss", "train_acc", "test_acc",
    "train_margin", "test_margin",
]
SUMMARY_FIELDS = [
    "seed", "path", "min_test_acc", "test_acc_barrier",
    "max_test_loss", "test_loss_barrier",
    "min_test_margin", "test_margin_barrier",
]


def process_seed(args, config, project_root, op_dir, seed, device):
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return [], []

    available_steps = discover_steps(ckpt_dir)
    start_step = resolve_step(available_steps, args.start, "start")
    end_step = resolve_step(available_steps, args.end, "end")
    mid_target = args.mid if args.mid is not None else (start_step + end_step) // 2
    mid_step = resolve_step(available_steps, mid_target, "mid")

    sd_start = load_sd(ckpt_dir, start_step)
    sd_mid = load_sd(ckpt_dir, mid_step)
    sd_end = load_sd(ckpt_dir, end_step)

    train_loader = make_loader(config.p, args.operation, "train", project_root, run_dir, args.batch_size)
    test_loader = make_loader(config.p, args.operation, "test", project_root, run_dir, args.batch_size)
    model = GrokkingTransformer(config).to(device)
    path_names = ["linear", "bezier"] if args.paths == "both" else [args.paths]
    alphas = torch.linspace(0, 1, args.num_points).tolist()

    print(
        f"seed={seed} run_dir={run_dir} start={start_step} mid={mid_step} end={end_step} "
        f"paths={path_names} points={args.num_points}"
    )

    rows = []
    for path_name in path_names:
        print(f"\nPath: {path_name}")
        for idx, alpha in enumerate(alphas):
            model.load_state_dict(path_state(path_name, sd_start, sd_mid, sd_end, alpha))
            train_loss, train_acc, train_margin = evaluate(model, train_loader, device)
            test_loss, test_acc, test_margin = evaluate(model, test_loader, device)
            row = {
                "seed": seed,
                "path": path_name,
                "alpha": float(alpha),
                "start_step": start_step,
                "mid_step": mid_step,
                "end_step": end_step,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_margin": train_margin,
                "test_margin": test_margin,
            }
            rows.append(row)
            if (idx + 1) % 100 == 0 or idx == 0 or idx == len(alphas) - 1:
                print(
                    f"  [{idx + 1}/{len(alphas)}] seed={seed} alpha={alpha:.3f} "
                    f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} test_margin={test_margin:.4f}"
                )

    summaries = [summarize_path(rows, path_name, seed) for path_name in path_names]
    summaries = [s for s in summaries if s is not None]
    out_dir = os.path.join(run_dir, "interpolation")
    write_csv(os.path.join(out_dir, "interpolation_paths.csv"), rows, PATH_FIELDS)
    write_csv(os.path.join(out_dir, "interpolation_summary.csv"), summaries, SUMMARY_FIELDS)
    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description="Linear and curved checkpoint interpolation")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--mid", type=int, default=None)
    parser.add_argument("--end", type=int, default=99000)
    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument("--paths", choices=["linear", "bezier", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    config = Config(args.operation)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = args.seeds if args.multi_seed else [args.seed]

    all_rows, all_summaries = [], []
    for seed in seeds:
        rows, summaries = process_seed(args, config, project_root, op_dir, seed, device)
        all_rows.extend(rows)
        all_summaries.extend(summaries)

    if args.multi_seed and all_rows:
        out_dir = os.path.join(project_root, "data", op_dir, "interpolation")
        write_csv(os.path.join(out_dir, "interpolation_paths_multiseed.csv"), all_rows, PATH_FIELDS)
        write_csv(os.path.join(out_dir, "interpolation_summary_multiseed.csv"), all_summaries, SUMMARY_FIELDS)
    elif all_rows:
        compat_csv = os.path.join(project_root, "data", op_dir, "interpolation.csv")
        write_csv(compat_csv, all_rows, PATH_FIELDS)


if __name__ == "__main__":
    main()
