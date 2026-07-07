#!/usr/bin/env python3
"""
Causal Mediation Analysis (CMA) via mean ablation of attention heads.

For each attention head, replaces its output with the dataset-average output
and measures the accuracy drop. Higher drop = more important head.

Outputs:
    data/{op}/cma_head_means.npz  - mean head attention outputs (keys: l{L}_h{H})
    data/{op}/cma.csv             - per-head ablation scores

Usage:
    python src/cma.py --operation add
    python src/cma.py --operation mul --step 50000
"""

import os
import csv
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from train import GrokkingTransformer, Config

OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}


def make_loader(p, operation, split, project_root, batch_size=512):
    """Create DataLoader for train/test/full split."""
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
    dataset = TensorDataset(inputs, labels_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def forward_manual(model, inputs, device, head_means=None, ablated_head=None):
    """Manual forward pass with optional mean ablation of one head.

    Replicates the model's forward pass while intercepting per-head attention
    outputs.  When *ablated_head* is set, that head's attn×V output is replaced
    by the pre-computed mean from *head_means*.

    Args:
        model: GrokkingTransformer in eval mode.
        inputs: [B, 3] token ids on *device*.
        head_means: dict (layer, head) -> [seq, d_k] tensor (on device), or None.
        ablated_head: (layer, head) tuple, or None.

    Returns:
        logits: [B, p]
        collected: dict (layer, head) -> [B, seq, d_k] (only when not ablating)
    """
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads
    d_k = model.blocks[0].attention.d_k

    x = model.embedding(inputs) + model.pos_encoding[:, :inputs.shape[1], :]
    collected = {} if ablated_head is None else None

    for l, block in enumerate(model.blocks):
        attn = block.attention
        B, seq_len, _ = x.shape

        Q = attn.W_q(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)
        K = attn.W_k(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)
        V = attn.W_v(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        head_out = torch.matmul(attn_weights, V)  # [B, heads, seq, d_k]

        # Collect or ablate
        if ablated_head is not None and l == ablated_head[0]:
            head_out[:, ablated_head[1], :, :] = head_means[ablated_head].unsqueeze(0)
        elif collected is not None:
            for h in range(num_heads):
                collected[(l, h)] = head_out[:, h, :, :]

        # Continue residual stream
        head_out = head_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        attn_result = attn.W_o(head_out)
        x = block.norm1(x + attn_result)
        ffn_out = block.ffn(x)
        x = block.norm2(x + ffn_out)

    logits = model.output(x[:, -1, :])
    return logits, collected


@torch.no_grad()
def collect_head_means(model, loader, device):
    """Collect per-head mean attn×V outputs across all samples.

    Returns:
        head_means: dict (layer, head) -> tensor [seq, d_k]
    """
    model.eval()
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads
    d_k = model.blocks[0].attention.d_k

    head_sums = {(l, h): torch.zeros(3, d_k) for l in range(num_layers) for h in range(num_heads)}
    total = 0

    for inputs, _ in loader:
        inputs = inputs.to(device)
        _, collected = forward_manual(model, inputs, device)

        for key, val in collected.items():
            head_sums[key] += val.sum(dim=0).cpu()
        total += inputs.shape[0]

    head_means = {k: v / total for k, v in head_sums.items()}
    return head_means


@torch.no_grad()
def evaluate_ablation(model, loader, device, head_means, ablated_head):
    """Evaluate accuracy with one head replaced by its mean output."""
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits, _ = forward_manual(model, inputs, device, head_means, ablated_head)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.shape[0]
    return correct / total


@torch.no_grad()
def evaluate_clean(model, loader, device):
    """Evaluate clean accuracy (no ablation)."""
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.shape[0]
    return correct / total


def process_step(step, model, config, device, project_root, op_dir):
    """Run CMA for a single checkpoint step."""
    p = config.p
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads

    ckpt_path = os.path.join(project_root, "data", op_dir, "checkpoints",
                             f"checkpoint_step_{step}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] checkpoint_step_{step}.pt not found")
        return None

    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(sd)
    model.eval()

    full_loader = make_loader(p, config.operation, "full", project_root)
    train_loader = make_loader(p, config.operation, "train", project_root)
    test_loader = make_loader(p, config.operation, "test", project_root)

    # 1. Collect mean head outputs
    head_means = collect_head_means(model, full_loader, device)

    means_dir = os.path.join(project_root, "data", op_dir, "cma")
    os.makedirs(means_dir, exist_ok=True)
    means_path = os.path.join(means_dir, f"cma_head_means_step_{step}.npz")
    np.savez(means_path, **{f"l{l}_h{h}": head_means[(l, h)].numpy()
                            for l in range(num_layers) for h in range(num_heads)})

    # 2. Clean accuracy
    train_acc_orig = evaluate_clean(model, train_loader, device)
    test_acc_orig = evaluate_clean(model, test_loader, device)

    # 3. Ablate each head
    rows = []
    for l in range(num_layers):
        for h in range(num_heads):
            name = f"l{l}_h{h}"
            train_acc = evaluate_ablation(model, train_loader, device, head_means, (l, h))
            test_acc = evaluate_ablation(model, test_loader, device, head_means, (l, h))
            rows.append({
                "step": step, "head": name, "layer": l, "head_idx": h,
                "train_acc_orig": f"{train_acc_orig:.6f}",
                "test_acc_orig": f"{test_acc_orig:.6f}",
                "train_acc_ablated": f"{train_acc:.6f}",
                "test_acc_ablated": f"{test_acc:.6f}",
                "train_acc_drop": f"{train_acc_orig - train_acc:.6f}",
                "test_acc_drop": f"{test_acc_orig - test_acc:.6f}",
            })

    print(f"  Step {step:6d} | train_acc={train_acc_orig:.4f}  test_acc={test_acc_orig:.4f}  |  "
          f"max_test_drop={max(float(r['test_acc_drop']) for r in rows):.4f}  "
          f"argmax={max(rows, key=lambda r: float(r['test_acc_drop']))['head']}")
    return rows


def main():
    DEFAULT_STEPS = [100, 1000, 3000, 5000, 10000, 30000, 50000, 90000]

    parser = argparse.ArgumentParser(description="CMA: mean-ablation of attention heads")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help=f"Checkpoint steps. Default: {DEFAULT_STEPS}")
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    steps = args.steps or DEFAULT_STEPS

    model = GrokkingTransformer(config).to(device)
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads

    print(f"operation={args.operation}  device={device}")
    print(f"layers={num_layers}  heads_per_layer={num_heads}  steps={steps}")

    all_results = []
    for idx, step in enumerate(steps):
        print(f"\n[{idx + 1}/{len(steps)}] Step {step} ...")
        rows = process_step(step, model, config, device, project_root, op_dir)
        if rows is not None:
            all_results.extend(rows)

    if not all_results:
        print("No results to save.")
        return

    # Save combined CSV
    csv_dir = os.path.join(project_root, "data", op_dir, "cma")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "cma.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
