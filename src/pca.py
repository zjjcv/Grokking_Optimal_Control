#!/usr/bin/env python3
"""
Per-checkpoint t-SNE projections for embedding, unembedding, and per-layer hidden states.

Preprocessing:
  - embedding / unembedding: center → remove PC1 → L2 normalize → t-SNE
  - hidden states (per layer): PCA→50 → t-SNE

Outputs:
    data/{op}/pca/tsne_step_{N}.npz

Usage:
    python src/pca.py --operation add
    python src/pca.py --operation mul --steps 1000 10000 90000
"""

import os
import sys
import argparse

import torch
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import GrokkingTransformer, Config

OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}

DEFAULT_STEPS = [100, 1000, 3000, 5000, 10000, 30000, 50000, 90000]


def compute_label(x, y, p, operation):
    if operation == "add":   return (x + y) % p
    elif operation == "sub": return (x - y) % p
    elif operation == "mul": return (x * y) % p
    elif operation == "div": return 0 if y == 0 else (x * pow(y, -1, p)) % p


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
    dataset = TensorDataset(inputs, labels_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def extract(model, loader, device, p):
    """Extract raw emb, unemb, per-layer hidden states, token_pos, labels."""
    model.eval()
    n_blocks = len(model.blocks)
    all_hidden = [[] for _ in range(n_blocks)]
    all_labels = []

    raw_emb = model.embedding.weight[:p, :].detach().cpu().float().numpy()
    raw_unemb = model.output.weight.detach().cpu().float().numpy()

    with torch.no_grad():
        for inputs, label_batch in loader:
            inputs = inputs.to(device)
            x = model.embedding(inputs) + model.pos_encoding[:, :inputs.shape[1], :]
            for i, block in enumerate(model.blocks):
                x = block(x)
                all_hidden[i].append(x.detach().cpu().numpy())
            all_labels.append(label_batch.numpy())

    hidden_states = [np.concatenate(h, axis=0) for h in all_hidden]
    labels = np.concatenate(all_labels, axis=0)
    return raw_emb, raw_unemb, hidden_states, labels


def pca2(M):
    """Center then project to first 2 principal components."""
    c = M - M.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return c @ Vt[:2].T


def preprocess_emb_unemb(M):
    """Center → L2 normalize."""
    c = M - M.mean(axis=0, keepdims=True)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-30)
    return c


def tsne_hidden(flat):
    """PCA→50 then t-SNE for a flattened hidden state matrix."""
    c = flat - flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    pca50 = c @ Vt[:50].T
    return TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(pca50)


def main():
    parser = argparse.ArgumentParser(description="Compute t-SNE projections for 4-panel plots")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    args = parser.parse_args()

    config = Config(args.operation)
    p = config.p
    device = "cuda" if torch.cuda.is_available() else "cpu"

    op_dir = OP_DIR[args.operation]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, "data", op_dir, "checkpoints")
    out_dir = os.path.join(project_root, "data", op_dir, "pca")
    os.makedirs(out_dir, exist_ok=True)

    steps = args.steps or list(DEFAULT_STEPS)
    loader = make_all_data(p, args.operation)

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

        raw_emb, raw_unemb, hidden_states, labels = extract(model, loader, device, p)
        n_samples = hidden_states[0].shape[0]
        d_model = hidden_states[0].shape[2]

        # emb/unemb: PCA top-2 (no normalization, circle lives in PCs)
        proj_emb = pca2(raw_emb)
        proj_unemb = pca2(raw_unemb)
        print("  emb/unemb PCA done")

        # emb/unemb: t-SNE with center+L2 (for comparison)
        proj_emb_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(
            preprocess_emb_unemb(raw_emb))
        proj_unemb_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(
            preprocess_emb_unemb(raw_unemb))
        print("  emb/unemb t-SNE done")

        proj_l0 = tsne_hidden(hidden_states[0].reshape(-1, d_model))
        print("  layer0 t-SNE done")

        proj_l1 = tsne_hidden(hidden_states[1].reshape(-1, d_model))
        print("  layer1 t-SNE done")

        token_pos = np.tile(np.arange(3), n_samples)

        out_path = os.path.join(out_dir, f"tsne_step_{step}.npz")
        np.savez(out_path,
                 proj_emb=proj_emb, proj_unemb=proj_unemb,
                 proj_emb_tsne=proj_emb_tsne, proj_unemb_tsne=proj_unemb_tsne,
                 proj_l0=proj_l0, proj_l1=proj_l1,
                 token_pos=token_pos, labels=labels)
        print(f"  saved -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
