#!/usr/bin/env python3
"""
Relate head structure to causal importance across training.

For each checkpoint, this script computes:
  1. OV-NFM structure score for each attention head.
  2. CMA score for each attention head, measured as relative test-accuracy drop
     under mean ablation.
  3. Pearson/Spearman correlation between the 8 head structure scores and the
     8 head CMA scores.

Single-seed layout:
    data/{op}/checkpoints/checkpoint_step_{step}.pt
    data/{op}/relation/relation_heads.csv
    data/{op}/relation/relation.csv

Multi-seed layouts supported:
    data/{op}/seed_{seed}/checkpoints/...
    data/seed_{seed}/{op}/checkpoints/...

In multi-seed mode, per-seed CSVs are written under each seed run directory and
an aggregate copy is written to data/{op}/relation/relation_multiseed.csv.
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, GrokkingTransformer
from nfm import compute_nfm
from cma import collect_head_means, evaluate_ablation, evaluate_clean


OP_DIRS = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}

DEFAULT_STEPS = [100, 1000, 3000, 5000, 10000, 30000, 50000, 90000]
DEFAULT_SEEDS = [0, 1, 2]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def primitive_root(p):
    if p == 2:
        return 1
    phi = p - 1
    factors = set()
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"No primitive root found for p={p}")


def dlog_nonzero_order(p):
    g = primitive_root(p)
    order = []
    val = 1
    for _ in range(p - 1):
        order.append(val)
        val = (val * g) % p
    return np.asarray(order, dtype=int)


def maybe_dlog_reorder(mat, operation, p):
    if operation not in ("mul", "div"):
        return mat
    order = dlog_nonzero_order(p)
    return mat[np.ix_(order, order)]


def diag_projection(mat):
    n = mat.shape[0]
    proj = np.zeros_like(mat, dtype=np.float64)
    rows = np.arange(n)
    for offset in range(n):
        cols = (rows + offset) % n
        proj[rows, cols] = mat[rows, cols].mean()
    return proj


def anti_projection(mat):
    n = mat.shape[0]
    proj = np.zeros_like(mat, dtype=np.float64)
    rows = np.arange(n)
    for total in range(n):
        cols = (total - rows) % n
        proj[rows, cols] = mat[rows, cols].mean()
    return proj


def ov_structure_score(mat, operation, p, eps=1e-12):
    """Return a larger-is-more-structured OV-NFM circulant alignment score."""
    m = np.asarray(maybe_dlog_reorder(mat, operation, p), dtype=np.float64)
    norm = np.linalg.norm(m, ord="fro")
    d_diag = np.linalg.norm(m - diag_projection(m), ord="fro") / (norm + eps)
    d_anti = np.linalg.norm(m - anti_projection(m), ord="fro") / (norm + eps)
    d_circ = min(d_diag, d_anti)
    return max(0.0, 1.0 - float(d_circ)), float(d_circ), float(d_diag), float(d_anti)


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


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


def load_pairs(project_root, run_dir, op_dir, split, operation, p):
    if split in ("train", "test"):
        seed_path = os.path.join(run_dir, f"{split}_data.json")
        legacy_path = os.path.join(project_root, "data", op_dir, f"{split}_data.json")
        path = seed_path if os.path.exists(seed_path) else legacy_path
        with open(path, "r") as f:
            return json.load(f)
    return [(x, y) for x in range(p) for y in range(p)
            if not (operation == "div" and y == 0)]


def make_loader(p, operation, split, project_root, run_dir, op_dir, batch_size=512):
    pairs = load_pairs(project_root, run_dir, op_dir, split, operation, p)
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


def load_checkpoint_model(config, ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model = GrokkingTransformer(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def process_step(args, config, project_root, op_dir, run_dir, seed, step, device):
    ckpt_path = os.path.join(run_dir, "checkpoints", f"checkpoint_step_{step}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] seed={seed} step={step}: checkpoint not found")
        return None, None

    model = load_checkpoint_model(config, ckpt_path, device)
    p = config.p
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads

    nfm = compute_nfm(model, p)
    full_loader = make_loader(p, args.operation, "full", project_root, run_dir, op_dir, args.batch_size)
    test_loader = make_loader(p, args.operation, "test", project_root, run_dir, op_dir, args.batch_size)

    head_means = {k: v.to(device) for k, v in collect_head_means(model, full_loader, device).items()}
    clean_acc = evaluate_clean(model, test_loader, device)

    head_rows = []
    structure_scores = []
    cma_scores = []
    for layer in range(num_layers):
        for head_idx in range(num_heads):
            head = f"l{layer}_h{head_idx}"
            component = f"ov_l{layer}_h{head_idx}"
            structure_score, d_circ, d_diag, d_anti = ov_structure_score(
                nfm[component], args.operation, p, args.eps
            )
            ablated_acc = evaluate_ablation(model, test_loader, device, head_means, (layer, head_idx))
            acc_drop = clean_acc - ablated_acc
            cma_score = acc_drop / max(clean_acc, args.eps)

            structure_scores.append(structure_score)
            cma_scores.append(cma_score)
            head_rows.append({
                "seed": seed,
                "step": step,
                "head": head,
                "layer": layer,
                "head_idx": head_idx,
                "structure_score": structure_score,
                "D_circ": d_circ,
                "D_diag": d_diag,
                "D_anti": d_anti,
                "test_acc_clean": clean_acc,
                "test_acc_ablated": ablated_acc,
                "test_acc_drop": acc_drop,
                "cma_score": cma_score,
            })

    pearson = safe_corr(structure_scores, cma_scores)
    spearman = safe_corr(rankdata(structure_scores), rankdata(cma_scores))
    step_row = {
        "seed": seed,
        "step": step,
        "pearson": pearson,
        "spearman": spearman,
        "num_heads": len(head_rows),
        "mean_structure_score": float(np.mean(structure_scores)),
        "mean_cma_score": float(np.mean(cma_scores)),
        "test_acc_clean": clean_acc,
    }

    print(
        f"  seed={seed} step={step:6d} | acc={clean_acc:.4f} "
        f"pearson={pearson:.4f} spearman={spearman:.4f}"
    )
    return head_rows, step_row


def format_row(row):
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = "" if np.isnan(value) else f"{value:.10f}"
        else:
            out[key] = value
    return out


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row))
    print(f"[OK] {path}")


def process_seed(args, config, project_root, op_dir, seed, device):
    set_seed(seed)
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return [], []

    relation_dir = os.path.join(run_dir, "relation")
    all_head_rows = []
    step_rows = []
    for step in args.steps:
        head_rows, step_row = process_step(args, config, project_root, op_dir, run_dir, seed, step, device)
        if head_rows is None:
            continue
        all_head_rows.extend(head_rows)
        step_rows.append(step_row)

    if all_head_rows:
        write_csv(
            os.path.join(relation_dir, "relation_heads.csv"),
            all_head_rows,
            list(all_head_rows[0].keys()),
        )
        write_csv(
            os.path.join(relation_dir, "relation.csv"),
            step_rows,
            list(step_rows[0].keys()),
        )
    return all_head_rows, step_rows


def process_operation(args, operation):
    config = Config(operation)
    args.operation = operation
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIRS[operation]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = args.seeds if args.multi_seed else [args.seed]

    print("\n" + "=" * 70)
    print(f"operation={operation} device={device} seeds={seeds} steps={args.steps}")
    print("=" * 70)

    all_heads = []
    all_steps = []
    for seed in seeds:
        head_rows, step_rows = process_seed(args, config, project_root, op_dir, seed, device)
        all_heads.extend(head_rows)
        all_steps.extend(step_rows)

    if args.multi_seed and all_steps:
        out_dir = os.path.join(project_root, "data", op_dir, "relation")
        write_csv(
            os.path.join(out_dir, "relation_heads_multiseed.csv"),
            all_heads,
            list(all_heads[0].keys()),
        )
        write_csv(
            os.path.join(out_dir, "relation_multiseed.csv"),
            all_steps,
            list(all_steps[0].keys()),
        )


def main():
    parser = argparse.ArgumentParser(description="Correlate OV-NFM structure with CMA head importance")
    parser.add_argument("--operation", choices=list(OP_DIRS.keys()) + ["all"], default="all")
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    operations = list(OP_DIRS.keys()) if args.operation == "all" else [args.operation]
    for operation in operations:
        process_operation(args, operation)


if __name__ == "__main__":
    main()
