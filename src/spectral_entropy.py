#!/usr/bin/env python3
"""
Spectral entropy of model parameters via squared singular-value energy.

For each checkpoint, computes spectral entropy for every parameter tensor:

    sigma = SVD(W)
    p_i = sigma_i^2 / sum_j sigma_j^2
    H = -sum_i p_i log(p_i)

For 1D tensors, entries are treated as singular-value analogues and the
distribution is proportional to squared entries.

Outputs:
    data/{op}/spectral_entropy.csv
    data/{op}/seed_{seed}/spectral_entropy.csv
    data/{op}/spectral_entropy_multiseed.csv

Usage:
    python src/spectral_entropy.py --operation add
    python src/spectral_entropy.py --operation mul --steps 1000 10000 90000
    python src/spectral_entropy.py --operation add --multi-seed --seeds 0 1 2
"""

import argparse
import csv
import os

import torch

from train import Config


OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}

DEFAULT_SEEDS = [0, 1, 2]


def discover_steps(ckpt_dir):
    """Discover all available checkpoint steps."""
    import re

    steps = []
    for filename in os.listdir(ckpt_dir):
        match = re.match(r"checkpoint_step_(\d+)\.pt", filename)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def spectral_entropy(tensor, device):
    """H = -sum p_i log(p_i), with p_i proportional to squared singular values."""
    tensor = tensor.detach().to(device=device, dtype=torch.float64)
    with torch.no_grad():
        if tensor.ndim <= 1:
            energy = tensor.square().reshape(-1)
        else:
            energy = torch.linalg.svdvals(tensor).square().reshape(-1)

        total = energy.sum()
        if total <= 0:
            return 0.0
        probabilities = energy / total
        probabilities = probabilities[probabilities > 0]
        entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.cpu())


def load_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def resolve_seed_run_dir(project_root, op_dir, seed, multi_seed):
    data_root = os.path.join(project_root, "data")
    legacy_dir = os.path.join(data_root, op_dir)
    if multi_seed:
        candidates = [
            os.path.join(legacy_dir, f"seed_{seed}"),
            os.path.join(data_root, f"seed_{seed}", op_dir),
        ]
    else:
        candidates = [legacy_dir]

    for run_dir in candidates:
        if os.path.isdir(os.path.join(run_dir, "checkpoints")):
            return run_dir
    return candidates[0]


def compute_seed(
    args,
    project_root,
    op_dir,
    seed,
    aggregate_writer=None,
    aggregate_param_names=None,
):
    device = torch.device(args.device)
    run_dir = resolve_seed_run_dir(
        project_root,
        op_dir,
        seed,
        args.multi_seed,
    )
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_csv = os.path.join(run_dir, "spectral_entropy.csv")

    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return aggregate_param_names

    steps = args.steps or discover_steps(ckpt_dir)

    param_names = None
    for step in steps:
        path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
        if os.path.exists(path):
            state_dict = load_state_dict(path)
            param_names = sorted(state_dict.keys())
            break

    if param_names is None:
        print(f"[SKIP] seed={seed}: no checkpoints found in {ckpt_dir}")
        return aggregate_param_names

    if aggregate_param_names is not None and param_names != aggregate_param_names:
        raise ValueError(
            f"Parameter names differ for seed={seed}; cannot create a shared "
            "multi-seed CSV."
        )

    os.makedirs(run_dir, exist_ok=True)
    print(
        f"operation={args.operation}  seed={seed}  "
        f"params={len(param_names)}  steps={len(steps)}  device={device}"
    )
    print(f"run_dir={run_dir}", flush=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + param_names)
        f.flush()

        for idx, step in enumerate(steps):
            path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            if not os.path.exists(path):
                print(f"  [SKIP] step {step}")
                continue

            state_dict = load_state_dict(path)

            row = [step]
            for name in param_names:
                row.append(
                    f"{spectral_entropy(state_dict[name], device):.6f}"
                )

            writer.writerow(row)
            if aggregate_writer is not None:
                aggregate_writer.writerow([seed] + row + [run_dir])
            f.flush()
            print(
                f"  [{idx + 1}/{len(steps)}] Step {step} done",
                flush=True,
            )

    print(f"Saved to: {out_csv}", flush=True)
    return param_names


def main():
    parser = argparse.ArgumentParser(description="Compute spectral entropy for model parameters")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Process multiple seed directories. Default seeds: 0 1 2.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for singular-value computation.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but CUDA is not available."
        )

    # Instantiate Config to preserve the same operation validation as training.
    Config(args.operation)
    op_dir = OP_DIR[args.operation]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seeds = args.seeds if args.multi_seed else [args.seed]

    if not args.multi_seed:
        compute_seed(args, project_root, op_dir, args.seed)
        return

    aggregate_path = os.path.join(
        project_root,
        "data",
        op_dir,
        "spectral_entropy_multiseed.csv",
    )
    os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)

    aggregate_file = None
    aggregate_writer = None
    aggregate_param_names = None
    try:
        for seed in seeds:
            if aggregate_file is None:
                run_dir = resolve_seed_run_dir(project_root, op_dir, seed, True)
                ckpt_dir = os.path.join(run_dir, "checkpoints")
                if not os.path.isdir(ckpt_dir):
                    print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
                    continue
                steps = args.steps or discover_steps(ckpt_dir)
                first_path = next(
                    (
                        os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
                        for step in steps
                        if os.path.exists(
                            os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
                        )
                    ),
                    None,
                )
                if first_path is None:
                    print(f"[SKIP] seed={seed}: no checkpoints found in {ckpt_dir}")
                    continue
                aggregate_param_names = sorted(load_state_dict(first_path).keys())
                aggregate_file = open(aggregate_path, "w", newline="")
                aggregate_writer = csv.writer(aggregate_file)
                aggregate_writer.writerow(
                    ["seed", "step"] + aggregate_param_names + ["run_dir"]
                )

            compute_seed(
                args,
                project_root,
                op_dir,
                seed,
                aggregate_writer,
                aggregate_param_names,
            )
    finally:
        if aggregate_file is not None:
            aggregate_file.close()
            print(f"Multi-seed spectral entropy saved to: {aggregate_path}")


if __name__ == "__main__":
    main()
