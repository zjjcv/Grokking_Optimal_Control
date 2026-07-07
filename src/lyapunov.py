#!/usr/bin/env python3
"""
Empirical Lyapunov diagnostic for Transformer hidden-state stabilization.

For checkpoint tau and layer ell, this computes

    V_hat_{ell,tau} = (1/N) sum_i ||h_{ell,tau}(x_i) - m_{y_i,ell,tau}||^2

and the normalized form

    V_tilde_{ell,tau}
      = sum_i ||h_i - m_{y_i}||^2
        / (sum_i ||h_i - h_bar||^2 + eps).

The contraction factor is

    rho_{ell,tau} = V_tilde_{ell+1,tau} / (V_tilde_{ell,tau} + eps).

Outputs:
    data/{op}/lyapunov/hidden_step_{step}.npz
    data/{op}/lyapunov/lyapunov.csv

Usage:
    python src/lyapunov.py --operation add
    python src/lyapunov.py --operation all --steps 1000 10000 90000
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, GrokkingTransformer


OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}

DEFAULT_STEPS = [100, 500, 1000, 3000, 5000, 10000, 30000, 50000, 90000]


def compute_label(x, y, p, operation):
    if operation == "add":
        return (x + y) % p
    if operation == "sub":
        return (x - y) % p
    if operation == "mul":
        return (x * y) % p
    if operation == "div":
        return 0 if y == 0 else (x * pow(y, -1, p)) % p
    raise ValueError(f"Unknown operation: {operation}")


def make_loader(p, operation, split, project_root, batch_size):
    if split in ("train", "test"):
        import json

        op_dir = OP_DIR[operation]
        data_path = os.path.join(project_root, "data", op_dir, f"{split}_data.json")
        with open(data_path, "r") as f:
            pairs = json.load(f)
    else:
        pairs = [(x, y) for x in range(p) for y in range(p)]

    xs, ys, labels = [], [], []
    for x, y in pairs:
        xs.append(x)
        ys.append(y)
        labels.append(compute_label(x, y, p, operation))

    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(inputs, labels_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def collect_final_token_hidden(model, loader, device):
    model.eval()
    n_layers = len(model.blocks)
    hidden_by_layer = [[] for _ in range(n_layers)]
    labels_all = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        h = model.embedding(inputs) + model.pos_encoding[:, :inputs.shape[1], :]
        for layer_idx, block in enumerate(model.blocks):
            h = block(h)
            hidden_by_layer[layer_idx].append(h[:, -1, :].detach().cpu().numpy())
        labels_all.append(labels.numpy())

    hidden_by_layer = [np.concatenate(parts, axis=0) for parts in hidden_by_layer]
    labels = np.concatenate(labels_all, axis=0).astype(np.int64)
    return hidden_by_layer, labels


def rule_centers(hidden, labels, p):
    d = hidden.shape[1]
    centers = np.zeros((p, d), dtype=np.float64)
    counts = np.bincount(labels, minlength=p).astype(np.float64)

    for c in range(p):
        if counts[c] > 0:
            centers[c] = hidden[labels == c].mean(axis=0)

    return centers, counts


def lyapunov_metrics(hidden, labels, p, eps):
    h = hidden.astype(np.float64, copy=False)
    centers, counts = rule_centers(h, labels, p)
    assigned = centers[labels]

    residual = h - assigned
    numerator = float(np.sum(residual * residual))
    v_hat = numerator / h.shape[0]

    h_bar = h.mean(axis=0, keepdims=True)
    centered = h - h_bar
    denominator = float(np.sum(centered * centered))
    v_tilde = numerator / (denominator + eps)

    return {
        "V_hat": v_hat,
        "V_tilde": v_tilde,
        "within_ss": numerator,
        "total_ss": denominator,
        "centers": centers,
        "counts": counts,
        "h_bar": h_bar.squeeze(0),
    }


def process_operation(operation, steps, split, batch_size, eps, save_hidden):
    config = Config(operation)
    p = config.p
    device = "cuda" if torch.cuda.is_available() else "cpu"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[operation]
    ckpt_dir = os.path.join(project_root, "data", op_dir, "checkpoints")
    out_dir = os.path.join(project_root, "data", op_dir, "lyapunov")
    os.makedirs(out_dir, exist_ok=True)

    loader = make_loader(p, operation, split, project_root, batch_size)
    n_layers = config.num_layers
    header = ["step", "split", "layer", "V_hat", "V_tilde", "within_ss", "total_ss", "rho_next"]
    out_csv = os.path.join(out_dir, "lyapunov.csv")

    print("=" * 60)
    print("Empirical Lyapunov diagnostic")
    print(f"operation  = {operation}")
    print(f"split      = {split}")
    print(f"samples    = {len(loader.dataset)}")
    print(f"layers     = {n_layers}")
    print(f"device     = {device}")
    print(f"output     = {out_csv}")
    print("=" * 60)

    model = GrokkingTransformer(config).to(device)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        for idx, step in enumerate(steps):
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] step {step}: checkpoint not found")
                continue

            state = torch.load(ckpt_path, map_location="cpu")
            state_dict = (
                state["model_state_dict"]
                if isinstance(state, dict) and "model_state_dict" in state
                else state
            )
            model.load_state_dict(state_dict)

            hidden_by_layer, labels = collect_final_token_hidden(model, loader, device)
            layer_results = [lyapunov_metrics(h, labels, p, eps) for h in hidden_by_layer]

            if save_hidden:
                save_dict = {"labels": labels}
                for layer_idx, h in enumerate(hidden_by_layer):
                    save_dict[f"h_l{layer_idx}"] = h.astype(np.float32)
                    save_dict[f"centers_l{layer_idx}"] = layer_results[layer_idx]["centers"].astype(np.float32)
                    save_dict[f"counts_l{layer_idx}"] = layer_results[layer_idx]["counts"]
                    save_dict[f"hbar_l{layer_idx}"] = layer_results[layer_idx]["h_bar"].astype(np.float32)
                hidden_path = os.path.join(out_dir, f"hidden_step_{step}.npz")
                np.savez_compressed(hidden_path, **save_dict)

            v_tilde = [r["V_tilde"] for r in layer_results]
            for layer_idx, result in enumerate(layer_results):
                rho = ""
                if layer_idx + 1 < len(layer_results):
                    rho = v_tilde[layer_idx + 1] / (v_tilde[layer_idx] + eps)

                writer.writerow([
                    step,
                    split,
                    layer_idx,
                    f"{result['V_hat']:.10f}",
                    f"{result['V_tilde']:.10f}",
                    f"{result['within_ss']:.10f}",
                    f"{result['total_ss']:.10f}",
                    f"{rho:.10f}" if rho != "" else "",
                ])
            f.flush()

            rho_msg = ""
            if len(v_tilde) > 1:
                rho_msg = f" | rho_0={v_tilde[1] / (v_tilde[0] + eps):.4f}"
            print(
                f"  [{idx + 1}/{len(steps)}] step={step:6d} "
                f"| V_tilde={', '.join(f'{v:.4f}' for v in v_tilde)}{rho_msg}"
            )

    print(f"Saved: {out_csv}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute empirical Lyapunov hidden-state diagnostic")
    parser.add_argument("--operation", choices=list(OP_DIR.keys()) + ["all"], default="add")
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--split", choices=["train", "test", "full"], default="full")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--no-save-hidden", action="store_true",
                        help="Only save lyapunov.csv; skip hidden/center npz files.")
    args = parser.parse_args()

    steps = args.steps or list(DEFAULT_STEPS)
    operations = list(OP_DIR.keys()) if args.operation == "all" else [args.operation]
    for operation in operations:
        process_operation(
            operation,
            steps,
            args.split,
            args.batch_size,
            args.eps,
            save_hidden=not args.no_save_hidden,
        )


if __name__ == "__main__":
    main()
