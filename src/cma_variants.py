#!/usr/bin/env python3
"""
CMA ablation variants: mean / zero / resample ablation of attention heads.

Mean ablation can push activations off-distribution.  This script adds two
variants and lets us check that head-importance rankings are robust to the
choice of ablation:
    - mean:     replace head attn*V output with its dataset mean (as cma.py)
    - zero:     replace with zeros
    - resample: replace with the same head's output on another randomly chosen
                input from the batch (stays exactly on-distribution)

Outputs:
    data/{op}/cma/ablation_variants.csv  - per-head, per-variant ablation scores

Usage:
    python src/cma_variants.py --operation add
    python src/cma_variants.py --operation mul --steps 50000 90000
"""

import argparse
import csv
import json
import os

import numpy as np
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

VARIANTS = ["mean", "zero", "resample"]


def make_loader(p, operation, split, project_root, batch_size=512):
    op_dir = OP_DIR[operation]
    if split in ("train", "test"):
        data_path = os.path.join(project_root, "data", op_dir, f"{split}_data.json")
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
        if operation == "add":     labels.append((x + y) % p)
        elif operation == "sub":   labels.append((x - y) % p)
        elif operation == "mul":   labels.append((x * y) % p)
        elif operation == "div":   labels.append((x * pow(y, -1, p)) % p)

    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, labels_t), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def forward_ablate(model, inputs, ablated_head=None, mode=None, head_means=None, generator=None):
    """Manual forward pass with optional ablation of one head.

    mode: None (collect head outputs), "mean", "zero", or "resample".
    Returns (logits, collected) where collected is populated only when mode is None.
    """
    num_heads = model.blocks[0].attention.num_heads
    d_k = model.blocks[0].attention.d_k

    x = model.embedding(inputs) + model.pos_encoding[:, :inputs.shape[1], :]
    collected = {} if mode is None else None

    for l, block in enumerate(model.blocks):
        attn = block.attention
        B, seq_len, _ = x.shape

        Q = attn.W_q(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)
        K = attn.W_k(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)
        V = attn.W_v(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        head_out = torch.matmul(attn_weights, V)  # [B, heads, seq, d_k]

        if mode is not None and ablated_head is not None and l == ablated_head[0]:
            h = ablated_head[1]
            if mode == "mean":
                head_out[:, h, :, :] = head_means[ablated_head].unsqueeze(0)
            elif mode == "zero":
                head_out[:, h, :, :] = 0.0
            elif mode == "resample":
                perm = torch.randperm(B, generator=generator, device=head_out.device)
                head_out[:, h, :, :] = head_out[perm, h, :, :]
        elif collected is not None:
            for h in range(num_heads):
                collected[(l, h)] = head_out[:, h, :, :]

        head_out = head_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        x = block.norm1(x + attn.W_o(head_out))
        x = block.norm2(x + block.ffn(x))

    logits = model.output(x[:, -1, :])
    return logits, collected


@torch.no_grad()
def collect_head_means(model, loader, device):
    model.eval()
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads
    d_k = model.blocks[0].attention.d_k

    head_sums = {(l, h): torch.zeros(3, d_k) for l in range(num_layers) for h in range(num_heads)}
    total = 0
    for inputs, _ in loader:
        inputs = inputs.to(device)
        _, collected = forward_ablate(model, inputs)
        for key, val in collected.items():
            head_sums[key] += val.sum(dim=0).cpu()
        total += inputs.shape[0]
    return {k: (v / total).to(device) for k, v in head_sums.items()}


@torch.no_grad()
def evaluate_variant(model, loader, device, ablated_head, mode, head_means, generator, trials):
    """Accuracy with one head ablated; resample averages over multiple trials."""
    model.eval()
    n_trials = trials if mode == "resample" else 1
    accs = []
    for _ in range(n_trials):
        correct, total = 0, 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _ = forward_ablate(model, inputs, ablated_head, mode, head_means, generator)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.shape[0]
        accs.append(correct / total)
    return float(np.mean(accs))


@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.shape[0]
    return correct / total


def process_step(step, model, config, device, project_root, op_dir, args):
    ckpt_path = os.path.join(project_root, "data", op_dir, "checkpoints",
                             f"checkpoint_step_{step}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] checkpoint_step_{step}.pt not found")
        return None

    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(sd)
    model.eval()

    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads
    p = config.p

    full_loader = make_loader(p, config.operation, "full", project_root)
    train_loader = make_loader(p, config.operation, "train", project_root)
    test_loader = make_loader(p, config.operation, "test", project_root)

    head_means = collect_head_means(model, full_loader, device)
    train_acc_orig = evaluate_clean(model, train_loader, device)
    test_acc_orig = evaluate_clean(model, test_loader, device)

    generator = torch.Generator(device=device).manual_seed(args.seed * 1000003 + step)

    rows = []
    for l in range(num_layers):
        for h in range(num_heads):
            for mode in VARIANTS:
                train_acc = evaluate_variant(model, train_loader, device, (l, h), mode,
                                             head_means, generator, args.resample_trials)
                test_acc = evaluate_variant(model, test_loader, device, (l, h), mode,
                                            head_means, generator, args.resample_trials)
                rows.append({
                    "step": step, "variant": mode, "head": f"l{l}_h{h}",
                    "layer": l, "head_idx": h,
                    "train_acc_orig": f"{train_acc_orig:.6f}",
                    "test_acc_orig": f"{test_acc_orig:.6f}",
                    "train_acc_ablated": f"{train_acc:.6f}",
                    "test_acc_ablated": f"{test_acc:.6f}",
                    "train_acc_drop": f"{train_acc_orig - train_acc:.6f}",
                    "test_acc_drop": f"{test_acc_orig - test_acc:.6f}",
                })

    by_variant = {}
    for mode in VARIANTS:
        drops = [float(r["test_acc_drop"]) for r in rows if r["variant"] == mode]
        by_variant[mode] = max(drops)
    print(f"  Step {step:6d} | clean train={train_acc_orig:.4f} test={test_acc_orig:.4f} | "
          f"max_test_drop: " + "  ".join(f"{m}={v:.4f}" for m, v in by_variant.items()))
    return rows


def main():
    DEFAULT_STEPS = [100, 1000, 3000, 5000, 10000, 30000, 50000, 90000]

    parser = argparse.ArgumentParser(description="CMA ablation variants (mean/zero/resample)")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help=f"Checkpoint steps. Default: {DEFAULT_STEPS}")
    parser.add_argument("--resample-trials", type=int, default=3,
                        help="Number of random permutations averaged for resample ablation.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    steps = args.steps or DEFAULT_STEPS

    model = GrokkingTransformer(config).to(device)
    print(f"operation={args.operation}  device={device}  steps={steps}")
    print(f"variants={VARIANTS}  resample_trials={args.resample_trials}")

    all_results = []
    for idx, step in enumerate(steps):
        print(f"\n[{idx + 1}/{len(steps)}] Step {step} ...")
        rows = process_step(step, model, config, device, project_root, op_dir, args)
        if rows is not None:
            all_results.extend(rows)

    if not all_results:
        print("No results to save.")
        return

    csv_dir = os.path.join(project_root, "data", op_dir, "cma")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "ablation_variants.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
