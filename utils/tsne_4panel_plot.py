#!/usr/bin/env python3
"""
2-panel t-SNE plot: Layer 0 hidden | Layer 1 hidden
One figure per checkpoint, colored by token position.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from train import GrokkingTransformer, Config

OP_DIR = {
    "add": ("x+y", "+"),
    "sub": ("x-y", "-"),
    "mul": ("x_mul_y", r"\times"),
    "div": ("x_div_y", r"\div"),
}

TOKEN_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
TOKEN_NAMES = [r"$x$", "op", r"$y$"]


def compute_label(x, y, p, operation):
    if operation == "add":   return (x + y) % p
    elif operation == "sub": return (x - y) % p
    elif operation == "mul": return (x * y) % p
    elif operation == "div":
        return 0 if y == 0 else (x * pow(y, -1, p)) % p


def make_all_data(p, operation, batch_size=512):
    xs, ys, labels = [], [], []
    for x in range(p):
        for y in range(p):
            if operation == "div" and y == 0:
                continue
            xs.append(x)
            ys.append(y)
            labels.append(compute_label(x, y, p, operation))
    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(inputs, labels_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), xs, ys


def extract_per_layer_hidden(model, loader, device):
    """Returns raw_emb, raw_unemb, [hidden_l0, hidden_l1], token_pos, labels."""
    model.eval()
    n_blocks = len(model.blocks)
    all_hidden = [[] for _ in range(n_blocks)]
    all_inputs = []
    all_labels = []

    raw_emb = model.embedding.weight[:model.p, :].detach().cpu().float().numpy()
    raw_unemb = model.output.weight.detach().cpu().float().numpy()

    with torch.no_grad():
        for inputs, label_batch in loader:
            inputs = inputs.to(device)
            x = model.embedding(inputs) + model.pos_encoding[:, :inputs.shape[1], :]
            for i, block in enumerate(model.blocks):
                x = block(x)
                all_hidden[i].append(x.detach().cpu().numpy())
            all_inputs.append(inputs.detach().cpu().numpy())
            all_labels.append(label_batch.numpy())

    hidden_states = [np.concatenate(h, axis=0) for h in all_hidden]
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return raw_emb, raw_unemb, hidden_states, all_inputs, all_labels


def tsne_normalize(M):
    """Center + L2 normalize each row."""
    c = M - M.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    return c / (norms + 1e-30)


def tsne_pca50_then_tsne(M):
    """PCA→50 then t-SNE for large matrices."""
    c = M - M.mean(axis=0)
    _U, S, Vt = np.linalg.svd(c, full_matrices=False)
    pca50 = c @ Vt[:50].T
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(pca50)


def plot_one(op_key, step, symbol, proj_l0, proj_l1,
             token_pos, labels, p, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Layer 0 hidden states ---
    ax = axes[0]
    for t in range(3):
        mask = token_pos == t
        ax.scatter(proj_l0[mask, 0], proj_l0[mask, 1],
                   c=TOKEN_COLORS[t], s=4, alpha=0.4,
                   label=TOKEN_NAMES[t], edgecolors="none", rasterized=True)
    ax.legend(loc="best", fontsize=8, markerscale=3)
    ax.set_title("Layer 0 hidden (t-SNE)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    # --- Layer 1 hidden states ---
    ax = axes[1]
    for t in range(3):
        mask = token_pos == t
        ax.scatter(proj_l1[mask, 0], proj_l1[mask, 1],
                   c=TOKEN_COLORS[t], s=4, alpha=0.4,
                   label=TOKEN_NAMES[t], edgecolors="none", rasterized=True)
    ax.legend(loc="best", fontsize=8, markerscale=3)
    ax.set_title("Layer 1 hidden (t-SNE)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"$x \\ {symbol} \\ y$ mod 97 — Step {step}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OP_DIR.keys()), default="add")
    args = parser.parse_args()

    config = Config(args.operation)
    p = config.p
    device = "cuda" if torch.cuda.is_available() else "cpu"
    op_dir, symbol = OP_DIR[args.operation]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, "data", op_dir, "checkpoints")
    out_dir = os.path.join(project_root, "results", op_dir, "pca")
    os.makedirs(out_dir, exist_ok=True)

    steps = [100, 1000, 5000, 10000, 30000, 50000, 90000]
    loader, xs_all, ys_all = make_all_data(p, args.operation)

    print(f"operation={args.operation}  p={p}  device={device}  dataset={len(loader.dataset)}")

    for idx, step in enumerate(steps):
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] step {step}")
            continue

        print(f"\n[{idx+1}/{len(steps)}] Step {step} ...", flush=True)

        state = torch.load(ckpt_path, map_location="cpu")
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model = GrokkingTransformer(config).to(device)
        model.load_state_dict(sd)

        raw_emb, raw_unemb, hidden_states, token_ids, labels = extract_per_layer_hidden(model, loader, device)
        n_samples = hidden_states[0].shape[0]
        d_model = hidden_states[0].shape[2]

        # t-SNE per layer (PCA→50 first)
        h0_flat = hidden_states[0].reshape(-1, d_model)
        proj_l0 = tsne_pca50_then_tsne(h0_flat)
        token_pos = np.tile(np.arange(3), n_samples)
        print("  layer0 t-SNE done")

        h1_flat = hidden_states[1].reshape(-1, d_model)
        proj_l1 = tsne_pca50_then_tsne(h1_flat)
        print("  layer1 t-SNE done")

        out_path = os.path.join(out_dir, f"tsne_hidden_step_{step}.pdf")
        plot_one(args.operation, step, symbol, proj_l0, proj_l1,
                 token_pos, labels, p, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
