#!/usr/bin/env python3
"""
Neural Feature Matrix (NFM) computation for Transformer grokking.

Single-seed legacy layout:
    data/{op}/checkpoints/
    data/{op}/nfm/

Multi-seed layouts supported:
    data/{op}/seed_{seed}/checkpoints/
    data/seed_{seed}/{op}/checkpoints/

Each seed writes NFM matrices under its own run directory.
"""

import argparse
import os
import re

import numpy as np
import torch

from train import Config, GrokkingTransformer


OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}
DEFAULT_STEPS = [100, 500, 1000, 3000, 5000, 10000, 30000, 50000, 90000]
DEFAULT_SEEDS = [0, 1, 2]


def compute_nfm(model, p):
    """Compute all NFM matrices for a given model checkpoint.

    Returns:
        dict mapping component name -> numpy array [p, p]
    """
    results = {}
    E = model.embedding.weight[:p, :].detach().cpu().float()
    results["embedding"] = (E @ E.T).numpy()

    U = model.output.weight.detach().cpu().float()
    results["unembedding"] = (U @ U.T).numpy()

    d_model = E.shape[1]
    num_heads = model.blocks[0].attention.num_heads
    d_k = d_model // num_heads

    for l, block in enumerate(model.blocks):
        attn = block.attention
        ffn = block.ffn

        W_in = ffn.linear1.weight.detach().cpu().float()
        results[f"mlp_in_l{l}"] = (E @ W_in.T @ W_in @ E.T).numpy()

        W_out = ffn.linear2.weight.detach().cpu().float()
        results[f"mlp_out_l{l}"] = (E @ W_out @ W_out.T @ E.T).numpy()

        W_q = attn.W_q.weight.detach().cpu().float()
        W_k = attn.W_k.weight.detach().cpu().float()
        W_v = attn.W_v.weight.detach().cpu().float()
        W_o = attn.W_o.weight.detach().cpu().float()

        for h in range(num_heads):
            W_Qh = W_q[h * d_k:(h + 1) * d_k, :]
            W_Kh = W_k[h * d_k:(h + 1) * d_k, :]
            results[f"qk_l{l}_h{h}"] = (E @ W_Qh.T @ W_Kh @ E.T).numpy()

            W_Vh = W_v[h * d_k:(h + 1) * d_k, :]
            W_Oh = W_o[:, h * d_k:(h + 1) * d_k]
            W_OV = W_Oh @ W_Vh
            results[f"ov_l{l}_h{h}"] = (E @ W_OV.T @ W_OV @ E.T).numpy()

    return results


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


def discover_steps(ckpt_dir):
    steps = []
    if not os.path.isdir(ckpt_dir):
        return steps
    for filename in os.listdir(ckpt_dir):
        match = re.match(r"checkpoint_step_(\d+)\.pt", filename)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def process_seed(args, config, project_root, op_dir, seed):
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_dir = os.path.join(run_dir, "nfm")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return

    steps = args.steps or DEFAULT_STEPS
    if args.all_checkpoints:
        steps = discover_steps(ckpt_dir)

    print("=" * 60)
    print("NFM computation")
    print(f"operation = {args.operation}")
    print(f"seed      = {seed}")
    print(f"run_dir   = {run_dir}")
    print(f"p         = {config.p}")
    print(f"steps     = {steps}")
    print(f"output    = {out_dir}")
    print("=" * 60)

    for idx, step in enumerate(steps):
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] seed={seed} step={step}: checkpoint not found")
            continue

        print(f"\n[{idx + 1}/{len(steps)}] seed={seed} step={step} ...", flush=True)
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

        model = GrokkingTransformer(config)
        model.load_state_dict(state_dict)
        model.eval()

        nfm = compute_nfm(model, config.p)
        for name, mat in nfm.items():
            np.save(os.path.join(out_dir, f"nfm_{name}_step_{step}.npy"), mat)

        print(f"  saved {len(nfm)} matrices, shapes: {set(m.shape for m in nfm.values())}")


def main():
    parser = argparse.ArgumentParser(description="Compute NFM for grokking checkpoints")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help=f"Checkpoint steps. Default: {DEFAULT_STEPS}")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Use every checkpoint_step_*.pt found in the checkpoint directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    config = Config(args.operation)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    seeds = args.seeds if args.multi_seed else [args.seed]
    for seed in seeds:
        process_seed(args, config, project_root, op_dir, seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
