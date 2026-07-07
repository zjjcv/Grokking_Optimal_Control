#!/usr/bin/env python3
"""
Hessian spectral probes for saved checkpoints.

For each checkpoint, computes exact full-batch (train split) Hessian probes
via Hessian-vector products:
    - top-k Hessian eigenvalues (Lanczos with full reorthogonalization)
    - minimum Ritz value (most negative curvature estimate)
    - trace(H) (Hutchinson, Rademacher probes)
    - trace(H^2) (Hutchinson, same probes)
    - participation-ratio effective rank: erank_pr = trace(H)^2 / trace(H^2)
    - top-eigenvalue effective rank:      erank_max = trace(H) / lambda_max

erank_pr / 2 serves as a Hessian-based degeneracy reference for the LLC:
in the regular quadratic case the learning coefficient equals rank(H) / 2,
and singular models satisfy lambda <= rank(H) / 2.

Single-seed legacy layout:
    data/{op}/checkpoints/checkpoint_step_{step}.pt  ->  data/{op}/hessian.csv

Multi-seed layouts supported (same as llc.py):
    data/{op}/seed_{seed}/checkpoints/  ->  data/{op}/seed_{seed}/hessian.csv
    data/seed_{seed}/{op}/checkpoints/  ->  data/seed_{seed}/{op}/hessian.csv
plus an aggregate data/{op}/hessian_multiseed.csv in multi-seed mode.

Usage:
    python src/hessian_probe.py --operation add
    python src/hessian_probe.py --operation add --multi-seed --seeds 0 1 2 --step-interval 100
    python src/hessian_probe.py --operation add --steps 100 1000 90000
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, GrokkingTransformer


OP_DIRS = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}

DEFAULT_SEEDS = [0, 1, 2]
TOP_K = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_tensors(p, operation, project_root, run_dir=None):
    legacy_path = os.path.join(project_root, "data", OP_DIRS[operation], "train_data.json")
    seed_path = os.path.join(run_dir, "train_data.json") if run_dir is not None else None
    data_path = seed_path if seed_path is not None and os.path.exists(seed_path) else legacy_path
    with open(data_path, "r") as f:
        pairs = json.load(f)

    labels = []
    for x, y in pairs:
        if operation == "add":
            labels.append((x + y) % p)
        elif operation == "sub":
            labels.append((x - y) % p)
        elif operation == "mul":
            labels.append((x * y) % p)
        elif operation == "div":
            labels.append(0 if y == 0 else (x * pow(y, -1, p)) % p)

    inputs = torch.tensor([[x, p, y] for x, y in pairs], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return inputs, labels_t


class HVPOperator:
    """Exact full-batch Hessian-vector products with a persistent grad graph."""

    def __init__(self, model, inputs, labels):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.num_params = sum(p.numel() for p in self.params)

        model.eval()  # disable dropout: deterministic loss surface
        logits = model(inputs)
        self.loss = F.cross_entropy(logits, labels)
        self.grads = torch.autograd.grad(self.loss, self.params, create_graph=True)
        self.flat_grad = torch.cat([g.reshape(-1) for g in self.grads])

    def apply(self, vec):
        """vec: flat tensor [num_params] -> H @ vec (flat)."""
        dot = torch.dot(self.flat_grad, vec)
        hv = torch.autograd.grad(dot, self.params, retain_graph=True)
        return torch.cat([h.reshape(-1) for h in hv]).detach()


def lanczos(hvp, dim, num_steps, device, generator):
    """Lanczos with full reorthogonalization; returns Ritz values (ascending)."""
    num_steps = min(num_steps, dim)
    q = torch.randn(dim, device=device, generator=generator)
    q /= q.norm()
    Q = torch.zeros(num_steps, dim, device=device)
    alphas, betas = [], []

    beta = 0.0
    q_prev = torch.zeros(dim, device=device)
    for j in range(num_steps):
        Q[j] = q
        w = hvp.apply(q)
        alpha = torch.dot(w, q).item()
        w = w - alpha * q - beta * q_prev
        # full reorthogonalization against all previous Lanczos vectors
        w = w - Q[: j + 1].T @ (Q[: j + 1] @ w)
        beta_next = w.norm().item()
        alphas.append(alpha)
        if j < num_steps - 1:
            if beta_next < 1e-10:
                break
            betas.append(beta_next)
            q_prev = q
            q = w / beta_next
            beta = beta_next

    T = np.diag(np.asarray(alphas))
    if betas:
        off = np.asarray(betas[: len(alphas) - 1])
        T += np.diag(off, k=1) + np.diag(off, k=-1)
    return np.sort(np.linalg.eigvalsh(T))


def hutchinson(hvp, dim, num_probes, device, generator):
    """Rademacher estimates of trace(H) and trace(H^2)."""
    trace_est, trace_sq_est = 0.0, 0.0
    for _ in range(num_probes):
        v = torch.randint(0, 2, (dim,), device=device, generator=generator, dtype=torch.float32)
        v = 2.0 * v - 1.0
        hv = hvp.apply(v)
        trace_est += torch.dot(v, hv).item()
        trace_sq_est += torch.dot(hv, hv).item()
    return trace_est / num_probes, trace_sq_est / num_probes


def checkpoint_step(filename):
    return int(filename.split("_")[-1].split(".")[0])


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


def probe_checkpoint(config, path, inputs, labels, device, args, probe_seed):
    state_dict = torch.load(path, map_location="cpu")["model_state_dict"]
    model = GrokkingTransformer(config).to(device)
    model.load_state_dict(state_dict)
    for p in model.parameters():
        p.requires_grad_(True)

    generator = torch.Generator(device=device).manual_seed(probe_seed)
    hvp = HVPOperator(model, inputs, labels)
    dim = hvp.num_params

    ritz = lanczos(hvp, dim, args.lanczos_steps, device, generator)
    top_eigs = ritz[::-1][:TOP_K]
    lambda_max = float(top_eigs[0])
    lambda_min = float(ritz[0])

    trace, trace_sq = hutchinson(hvp, dim, args.hutchinson_probes, device, generator)
    erank_pr = trace * trace / trace_sq if trace_sq > 0 else 0.0
    erank_max = trace / lambda_max if lambda_max > 0 else 0.0

    return {
        "loss": float(hvp.loss.item()),
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "trace": trace,
        "trace_sq": trace_sq,
        "erank_pr": erank_pr,
        "erank_max": erank_max,
        "top_eigs": [float(v) for v in top_eigs],
    }


def compute_seed_hessian(args, config, project_root, op_dir, seed, device, aggregate_writer=None):
    set_seed(seed)
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    output_csv = os.path.join(run_dir, "hessian.csv")

    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return

    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_step_") and f.endswith(".pt")],
        key=checkpoint_step,
    )
    if args.steps is not None:
        wanted = set(args.steps)
        ckpt_files = [f for f in ckpt_files if checkpoint_step(f) in wanted]
    elif args.step_interval > 0:
        ckpt_files = [f for f in ckpt_files if checkpoint_step(f) % args.step_interval == 0]

    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")
    print(f"seed={seed}  device={device}  lanczos={args.lanczos_steps}  probes={args.hutchinson_probes}")

    inputs, labels = load_train_tensors(config.p, args.operation, project_root, run_dir=run_dir)
    inputs, labels = inputs.to(device), labels.to(device)

    base_cols = ["step", "seed", "loss", "lambda_max", "lambda_min",
                 "trace", "trace_sq", "erank_pr", "erank_max"]
    eig_cols = [f"lambda_top_{i + 1}" for i in range(TOP_K)]

    os.makedirs(run_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(base_cols + eig_cols)
        f.flush()

        for idx, ckpt_file in enumerate(ckpt_files):
            step = checkpoint_step(ckpt_file)
            path = os.path.join(ckpt_dir, ckpt_file)
            # probe randomness reproducible per (seed, step)
            res = probe_checkpoint(config, path, inputs, labels, device, args,
                                   probe_seed=seed * 1000003 + step)

            row = [step, seed, f"{res['loss']:.8f}",
                   f"{res['lambda_max']:.8f}", f"{res['lambda_min']:.8f}",
                   f"{res['trace']:.8f}", f"{res['trace_sq']:.8f}",
                   f"{res['erank_pr']:.6f}", f"{res['erank_max']:.6f}"]
            eig_row = [f"{v:.8f}" for v in res["top_eigs"]]
            eig_row += ["nan"] * (TOP_K - len(eig_row))
            writer.writerow(row + eig_row)
            f.flush()

            if aggregate_writer is not None:
                aggregate_writer.writerow(row + eig_row + [run_dir])

            print(
                f"  [{idx + 1}/{len(ckpt_files)}] seed={seed} step={step:6d} | "
                f"lam_max={res['lambda_max']:.4f} trace={res['trace']:.4f} "
                f"erank_pr={res['erank_pr']:.2f}"
            )

    print(f"Hessian probe data saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Hessian spectral probes for grokking checkpoints")
    parser.add_argument("--operation", type=str, default="add", choices=list(OP_DIRS))
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Explicit checkpoint steps (overrides --step-interval).")
    parser.add_argument("--step-interval", type=int, default=0,
                        help="Only process checkpoints whose step is a multiple of this value (0=all).")
    parser.add_argument("--lanczos-steps", type=int, default=64,
                        help="Lanczos iterations (Ritz values reported: top-20 and min).")
    parser.add_argument("--hutchinson-probes", type=int, default=32,
                        help="Rademacher probes for trace(H) and trace(H^2).")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run for multiple seed directories. Default seeds: 0 1 2.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Single-seed mode probe seed.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seeds used when --multi-seed is enabled.")
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIRS[args.operation]
    seeds = args.seeds if args.multi_seed else [args.seed]

    aggregate_file = None
    aggregate_writer = None
    aggregate_path = os.path.join(project_root, "data", op_dir, "hessian_multiseed.csv")
    if args.multi_seed:
        os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)
        aggregate_file = open(aggregate_path, "w", newline="")
        aggregate_writer = csv.writer(aggregate_file)
        base_cols = ["step", "seed", "loss", "lambda_max", "lambda_min",
                     "trace", "trace_sq", "erank_pr", "erank_max"]
        eig_cols = [f"lambda_top_{i + 1}" for i in range(TOP_K)]
        aggregate_writer.writerow(base_cols + eig_cols + ["run_dir"])

    try:
        for seed in seeds:
            compute_seed_hessian(args, config, project_root, op_dir, seed, device, aggregate_writer)
    finally:
        if aggregate_file is not None:
            aggregate_file.close()
            print(f"\nMulti-seed Hessian probe data saved to: {aggregate_path}")


if __name__ == "__main__":
    main()
