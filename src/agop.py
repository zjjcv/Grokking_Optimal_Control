#!/usr/bin/env python3
"""
Paper-style AGOP computation for Transformer grokking experiments.

This script computes the Average Gradient Outer Product (AGOP):

    G(f) = (1/N) sum_i J_f(x_i)^T J_f(x_i)

where J_f is the Jacobian of all output logits with respect to the
one-hot input vector [onehot(x), onehot(y)] in R^{2p}.

For Transformer token inputs, token ids are discrete. We therefore use
the embedding chain rule:

    d z_k / d onehot(x) = E_num @ d z_k / d emb_x
    d z_k / d onehot(y) = E_num @ d z_k / d emb_y

where E_num is the numeric-token embedding matrix of shape [p, d_model].

The output AGOP has shape [2p, 2p]:

    AGOP = [[G_xx, G_xy],
            [G_yx, G_yy]]

This is the closest analogue of the AGOP used in the RFM/grokking paper.

Outputs:
    data/{op}/agop_paper/agop_paper_step_{step}_raw.csv/.npy
    data/{op}/agop_paper/agop_paper_step_{step}_sqrt.csv/.npy

Usage:
    python src/agop_paper.py --operation add --steps 50000 90000
    python src/agop_paper.py --operation add --split full --steps 90000
    python src/agop_paper.py --operation add --center-logits --steps 90000
"""

import os
import sys
import csv
import json
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import GrokkingTransformer, Config


OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x*y",
    "div": "x_div_y",
}


def make_dataloader(p, operation, batch_size=512, split="train"):
    """
    split="train": use data/{op}/train_data.json
    split="full":  use all possible input pairs.

    For div, y=0 is excluded.
    """
    op_dir = OP_DIR[operation]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if split == "train":
        data_path = os.path.join(project_root, "data", op_dir, "train_data.json")
        with open(data_path, "r") as f:
            pairs = json.load(f)

    elif split == "full":
        pairs = []
        for x in range(p):
            for y in range(p):
                if operation == "div" and y == 0:
                    continue
                pairs.append([x, y])
    else:
        raise ValueError(f"Unknown split: {split}")

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
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # Keep the same input format as training: [x, p, y]
    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(inputs, labels_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def forward_from_embedding(model, emb_var):
    """
    Forward pass starting from differentiable embeddings.

    emb_var: [B, 3, d_model]
    return: logits [B, p]
    """
    h = emb_var + model.pos_encoding[:, : emb_var.shape[1], :]

    for block in model.blocks:
        h = block(h)

    h_last = h[:, -1, :]
    logits = model.output(h_last)
    return logits


def psd_sqrt(M, eps=1e-10):
    """
    Matrix square root for a symmetric PSD matrix.
    """
    M = 0.5 * (M + M.T)
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=eps)
    return (evecs * torch.sqrt(evals)) @ evecs.T


def center_2p_matrix(M, p):
    """
    Remove constant token mode separately from x and y blocks.

    This is not part of the basic paper-style AGOP.
    Use only as an optional diagnostic.
    """
    device = M.device
    dtype = M.dtype

    I = torch.eye(p, device=device, dtype=dtype)
    one = torch.ones(p, p, device=device, dtype=dtype) / p
    P = I - one

    B = torch.zeros(2 * p, 2 * p, device=device, dtype=dtype)
    B[:p, :p] = P
    B[p:, p:] = P

    return B @ M @ B.T


def compute_paper_agop(
    model,
    loader,
    p,
    device,
    center_logits=False,
    center_token=False,
    dtype=torch.float64,
):
    """
    Compute paper-style AGOP:

        G = (1/N) sum_i J_i^T J_i

    where J_i is the Jacobian of all logits with respect to
    [onehot(x), onehot(y)].

    The one-hot Jacobian is obtained by chain rule through embedding:

        d z_k / d onehot(x) = E_num @ d z_k / d emb_x
        d z_k / d onehot(y) = E_num @ d z_k / d emb_y

    Returns:
        AGOP_raw:  [2p, 2p]
        AGOP_sqrt: [2p, 2p]
        total_samples
    """
    model.eval()
    model.to(device)

    E_num = model.embedding.weight[:p, :].detach().to(device)  # [p, d_model]

    G_xx = torch.zeros(p, p, device=device, dtype=dtype)
    G_yy = torch.zeros(p, p, device=device, dtype=dtype)
    G_xy = torch.zeros(p, p, device=device, dtype=dtype)

    total_samples = 0

    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.to(device)
        B = inputs.shape[0]

        emb = model.embedding(inputs)  # [B, 3, d_model]
        emb_var = emb.detach().requires_grad_(True)

        logits = forward_from_embedding(model, emb_var)  # [B, p]

        if center_logits:
            logits = logits - logits.mean(dim=1, keepdim=True)

        # Multi-output AGOP:
        # J^T J = sum_k grad(z_k) grad(z_k)^T
        for k in range(p):
            retain = k < p - 1

            grad_k = torch.autograd.grad(
                logits[:, k].sum(),
                emb_var,
                retain_graph=retain,
                create_graph=False,
            )[0]  # [B, 3, d_model]

            gx_emb = grad_k[:, 0, :]  # [B, d_model]
            gy_emb = grad_k[:, 2, :]  # [B, d_model]

            # Chain rule to token one-hot basis:
            # gx_tok[b, a] = <d z_k / d emb_x[b], E_num[a]>
            gx_tok = gx_emb @ E_num.T  # [B, p]
            gy_tok = gy_emb @ E_num.T  # [B, p]

            gx_tok = gx_tok.to(dtype)
            gy_tok = gy_tok.to(dtype)

            G_xx += gx_tok.T @ gx_tok
            G_yy += gy_tok.T @ gy_tok
            G_xy += gx_tok.T @ gy_tok

        total_samples += B

        if total_samples % 2000 == 0:
            print(f"    {total_samples} samples processed", flush=True)

    if total_samples == 0:
        raise RuntimeError("No samples were processed.")

    G_xx /= total_samples
    G_yy /= total_samples
    G_xy /= total_samples

    AGOP_raw = torch.zeros(2 * p, 2 * p, device=device, dtype=dtype)
    AGOP_raw[:p, :p] = G_xx
    AGOP_raw[p:, p:] = G_yy
    AGOP_raw[:p, p:] = G_xy
    AGOP_raw[p:, :p] = G_xy.T

    AGOP_raw = 0.5 * (AGOP_raw + AGOP_raw.T)

    if center_token:
        AGOP_raw = center_2p_matrix(AGOP_raw, p)
        AGOP_raw = 0.5 * (AGOP_raw + AGOP_raw.T)

    AGOP_sqrt = psd_sqrt(AGOP_raw)

    print("\n[BLOCK STATS]")
    print(f"G_xx abs max: {G_xx.abs().max().item():.6e} | fro: {torch.linalg.norm(G_xx).item():.6e}")
    print(f"G_yy abs max: {G_yy.abs().max().item():.6e} | fro: {torch.linalg.norm(G_yy).item():.6e}")
    print(f"G_xy abs max: {G_xy.abs().max().item():.6e} | fro: {torch.linalg.norm(G_xy).item():.6e}")
    print(
        "xy / xx fro : "
        f"{(torch.linalg.norm(G_xy) / (torch.linalg.norm(G_xx) + 1e-12)).item():.6e}"
    )
    print(
        "xy / yy fro : "
        f"{(torch.linalg.norm(G_xy) / (torch.linalg.norm(G_yy) + 1e-12)).item():.6e}"
    )

    return (
        AGOP_raw.detach().cpu().numpy(),
        AGOP_sqrt.detach().cpu().numpy(),
        total_samples,
    )


def save_csv(path, matrix):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow([f"{v:.8e}" for v in row])


def save_matrix_pair(out_dir, prefix, raw, sqrt):
    raw_csv = os.path.join(out_dir, f"{prefix}_raw.csv")
    sqrt_csv = os.path.join(out_dir, f"{prefix}_sqrt.csv")

    raw_npy = os.path.join(out_dir, f"{prefix}_raw.npy")
    sqrt_npy = os.path.join(out_dir, f"{prefix}_sqrt.npy")

    save_csv(raw_csv, raw)
    save_csv(sqrt_csv, sqrt)

    np.save(raw_npy, raw)
    np.save(sqrt_npy, sqrt)

    return raw_csv, sqrt_csv, raw_npy, sqrt_npy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--operation",
        default="add",
        choices=["add", "sub", "mul", "div"],
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Checkpoint steps. Default: representative points.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "full"],
        help=(
            "Paper-style AGOP is usually computed on training samples. "
            "Use full only if you want the whole Z_p^2 grid."
        ),
    )

    parser.add_argument(
        "--center-logits",
        action="store_true",
        help=(
            "Optional diagnostic: subtract mean logit before computing AGOP. "
            "Default False to match raw paper-style multi-output AGOP."
        ),
    )

    parser.add_argument(
        "--center-token",
        action="store_true",
        help=(
            "Optional diagnostic: remove constant token mode from the final 2p x 2p AGOP. "
            "Default False."
        ),
    )

    parser.add_argument(
        "--float64",
        action="store_true",
        help="Use float64 accumulation and eigendecomposition. Default True-like behavior.",
    )

    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = config.p

    op_dir = OP_DIR[args.operation]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, "data", op_dir, "checkpoints")
    out_dir = os.path.join(project_root, "data", op_dir, "agop_paper")
    os.makedirs(out_dir, exist_ok=True)

    if args.steps is None:
        args.steps = [100, 500, 1000, 3000, 5000, 10000, 30000, 50000, 90000]

    loader = make_dataloader(
        p=p,
        operation=args.operation,
        batch_size=args.batch_size,
        split=args.split,
    )

    dtype = torch.float64 if args.float64 or True else torch.float32

    print("=" * 80)
    print("Paper-style AGOP computation")
    print(f"operation      = {args.operation}")
    print(f"split          = {args.split}")
    print(f"dataset size   = {len(loader.dataset)}")
    print(f"p              = {p}")
    print(f"device         = {device}")
    print(f"batch_size     = {args.batch_size}")
    print(f"center_logits  = {args.center_logits}")
    print(f"center_token   = {args.center_token}")
    print(f"dtype          = {dtype}")
    print("=" * 80)

    for idx, step in enumerate(args.steps):
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")

        if not os.path.exists(ckpt_path):
            print(f"[SKIP] step {step}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n[{idx + 1}/{len(args.steps)}] Step {step} ...", flush=True)

        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state

        model = GrokkingTransformer(config).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        agop_raw, agop_sqrt, n_eff = compute_paper_agop(
            model=model,
            loader=loader,
            p=p,
            device=device,
            center_logits=args.center_logits,
            center_token=args.center_token,
            dtype=dtype,
        )

        extra = ""
        if args.center_logits:
            extra += "_centerlogits"
        if args.center_token:
            extra += "_centertoken"

        prefix = f"agop_paper{extra}_step_{step}"

        raw_csv, sqrt_csv, raw_npy, sqrt_npy = save_matrix_pair(
            out_dir=out_dir,
            prefix=prefix,
            raw=agop_raw,
            sqrt=agop_sqrt,
        )

        print(f"  effective samples = {n_eff}")
        print(f"  raw  range = [{agop_raw.min():.4e}, {agop_raw.max():.4e}]")
        print(f"  sqrt range = [{agop_sqrt.min():.4e}, {agop_sqrt.max():.4e}]")
        print(f"  saved raw csv  -> {raw_csv}")
        print(f"  saved sqrt csv -> {sqrt_csv}")
        print(f"  saved raw npy  -> {raw_npy}")
        print(f"  saved sqrt npy -> {sqrt_npy}")

    print("\nDone.")


if __name__ == "__main__":
    main()