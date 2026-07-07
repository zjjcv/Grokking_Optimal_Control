#!/usr/bin/env python3
"""
Average Gradient Outer Product (AGOP) computation for Transformer grokking.

Single-seed legacy layout:
    data/{op}/checkpoints/
    data/{op}/agop/

Multi-seed layouts supported:
    data/{op}/seed_{seed}/checkpoints/
    data/seed_{seed}/{op}/checkpoints/

Each seed writes AGOP matrices under its own run directory.
"""

import argparse
import json
import os
import re
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


def make_dataloader(p, operation, project_root, run_dir, batch_size=512, split="full"):
    op_dir = OP_DIR[operation]
    if split in ("train", "test"):
        seed_path = os.path.join(run_dir, f"{split}_data.json")
        legacy_path = os.path.join(project_root, "data", op_dir, f"{split}_data.json")
        data_path = seed_path if os.path.exists(seed_path) else legacy_path
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

    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, labels_t), batch_size=batch_size, shuffle=False)


def forward_from_embedding(model, emb_var):
    h = emb_var + model.pos_encoding[:, :emb_var.shape[1], :]
    for block in model.blocks:
        h = block(h)
    return model.output(h[:, -1, :])


def compute_agop(model, loader, p, device, dtype=torch.float64):
    model.eval()
    E_num = model.embedding.weight[:p, :].detach().to(device)
    G_xx = torch.zeros(p, p, device=device, dtype=dtype)
    G_yy = torch.zeros(p, p, device=device, dtype=dtype)
    G_xy = torch.zeros(p, p, device=device, dtype=dtype)
    total_samples = 0

    for inputs, _ in loader:
        inputs = inputs.to(device)
        batch_size = inputs.shape[0]
        emb = model.embedding(inputs)
        emb_var = emb.detach().requires_grad_(True)
        logits = forward_from_embedding(model, emb_var)

        for k in range(p):
            retain = k < p - 1
            grad_k = torch.autograd.grad(
                logits[:, k].sum(),
                emb_var,
                retain_graph=retain,
                create_graph=False,
            )[0]
            gx_emb = grad_k[:, 0, :]
            gy_emb = grad_k[:, 2, :]
            gx_tok = (gx_emb @ E_num.T).to(dtype)
            gy_tok = (gy_emb @ E_num.T).to(dtype)
            G_xx += gx_tok.T @ gx_tok
            G_yy += gy_tok.T @ gy_tok
            G_xy += gx_tok.T @ gy_tok

        total_samples += batch_size
        if total_samples % 2000 == 0:
            print(f"    {total_samples} samples processed", flush=True)

    if total_samples == 0:
        raise RuntimeError("No samples were processed.")

    G_xx /= total_samples
    G_yy /= total_samples
    G_xy /= total_samples
    agop = torch.zeros(2 * p, 2 * p, device=device, dtype=dtype)
    agop[:p, :p] = G_xx
    agop[p:, p:] = G_yy
    agop[:p, p:] = G_xy
    agop[p:, :p] = G_xy.T
    return agop.detach().cpu().numpy(), total_samples


def discover_steps(ckpt_dir):
    steps = []
    if not os.path.isdir(ckpt_dir):
        return steps
    for filename in os.listdir(ckpt_dir):
        match = re.match(r"checkpoint_step_(\d+)\.pt", filename)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def process_seed(args, config, project_root, op_dir, seed, device):
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_dir = os.path.join(run_dir, "agop")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return

    steps = args.steps or DEFAULT_STEPS
    if args.all_checkpoints:
        steps = discover_steps(ckpt_dir)

    loader = make_dataloader(
        p=config.p,
        operation=args.operation,
        project_root=project_root,
        run_dir=run_dir,
        batch_size=args.batch_size,
        split=args.split,
    )

    print("=" * 60)
    print("AGOP computation")
    print(f"operation    = {args.operation}")
    print(f"seed         = {seed}")
    print(f"run_dir      = {run_dir}")
    print(f"split        = {args.split}")
    print(f"dataset size = {len(loader.dataset)}")
    print(f"p            = {config.p}")
    print(f"device       = {device}")
    print(f"steps        = {steps}")
    print(f"output       = {out_dir}")
    print("=" * 60)

    for idx, step in enumerate(steps):
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] seed={seed} step={step}: checkpoint not found")
            continue

        print(f"\n[{idx + 1}/{len(steps)}] seed={seed} step={step} ...", flush=True)
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

        model = GrokkingTransformer(config).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        agop, _ = compute_agop(model, loader, config.p, device)
        out_path = os.path.join(out_dir, f"agop_step_{step}.npy")
        np.save(out_path, agop)
        print(f"  shape = {agop.shape}")
        print(f"  range = [{agop.min():.4e}, {agop.max():.4e}]")
        print(f"  saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute raw AGOP for grokking checkpoints")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help=f"Checkpoint steps. Default: {DEFAULT_STEPS}")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Use every checkpoint_step_*.pt found in the checkpoint directory.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--split", default="full", choices=["train", "test", "full"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    seeds = args.seeds if args.multi_seed else [args.seed]
    for seed in seeds:
        process_seed(args, config, project_root, op_dir, seed, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
