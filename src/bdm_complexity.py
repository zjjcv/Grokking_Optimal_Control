#!/usr/bin/env python3
"""
BDM/CTM algorithmic-complexity proxy for Transformer grokking.

For each checkpoint, evaluate the model on the full modular input space and compute:
  1. BDM of prediction-table bit planes.
  2. BDM of correctness table.
  3. Shannon entropy baselines from pybdm.
  4. For mul/div, BDM after discrete-log reindexing of both inputs and outputs.

This is not exact Kolmogorov complexity. It is a BDM/CTM-based computable proxy.

Usage:
    python src/bdm_complexity.py --operation add
    python src/bdm_complexity.py --operation mul --steps 1000 10000 90000
    python src/bdm_complexity.py --operation add --multi-seed --seeds 0 1 2
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import GrokkingTransformer, Config

from pybdm import BDM
from pybdm.partitions import PartitionIgnore


OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}

DEFAULT_SEEDS = [0, 1, 2]
CSV_COLUMNS = [
    "step",
    "bdm_pred_per_cell",
    "entropy_pred_per_cell",
    "bdm_correct_per_cell",
    "entropy_correct_per_cell",
    "bdm_pred_dlog_per_cell",
    "entropy_pred_dlog_per_cell",
    "bdm_correct_dlog_per_cell",
    "entropy_correct_dlog_per_cell",
]


# ============================================================
# Modular arithmetic data
# ============================================================

def compute_label(x, y, p, operation):
    if operation == "add":
        return (x + y) % p
    if operation == "sub":
        return (x - y) % p
    if operation == "mul":
        return (x * y) % p
    if operation == "div":
        if y == 0:
            return 0
        return (x * pow(y, -1, p)) % p
    raise ValueError(f"Unknown operation: {operation}")


def make_full_loader(p, operation, batch_size=512):
    """
    Full modular input space.

    For consistency with the training script, division includes y=0
    and assigns label 0.  For discrete-log analysis of div, the
    nonzero subtable is used later.
    """
    xs, ys, labels = [], [], []

    for x in range(p):
        for y in range(p):
            xs.append(x)
            ys.append(y)
            labels.append(compute_label(x, y, p, operation))

    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(inputs, labels_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def prediction_table(model, loader, p, device):
    """
    Return prediction table P and true table Y, both [p, p].

    P[a,b] = argmax_c f_theta(a,b)_c
    Y[a,b] = ground-truth modular label
    """
    model.eval()

    preds_all = []
    labels_all = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=-1).detach().cpu().numpy()

        preds_all.append(preds)
        labels_all.append(labels.numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    P = preds_all.reshape(p, p).astype(np.uint8)
    Y = labels_all.reshape(p, p).astype(np.uint8)

    return P, Y


# ============================================================
# Discrete-log reindexing for multiplication/division
# ============================================================

def primitive_root_mod_p(p):
    """
    Find a primitive root modulo prime p.
    """
    factors = []
    x = p - 1
    q = 2

    while q * q <= x:
        if x % q == 0:
            factors.append(q)
            while x % q == 0:
                x //= q
        q += 1

    if x > 1:
        factors.append(x)

    for g in range(2, p):
        ok = True
        for q in factors:
            if pow(g, (p - 1) // q, p) == 1:
                ok = False
                break
        if ok:
            return g

    raise RuntimeError(f"No primitive root found for p={p}")


def dlog_maps(p):
    """
    Return primitive root g, order list [g^0, ..., g^(p-2)],
    and log map z -> k such that z = g^k mod p.
    """
    g = primitive_root_mod_p(p)

    order = []
    log = {}

    x = 1
    for k in range(p - 1):
        order.append(x)
        log[x] = k
        x = (x * g) % p

    return g, order, log


def multiplicative_table_dlog(P, p, zero_symbol=None):
    """
    Convert a prediction or label table to discrete-log coordinates
    over Z_p^*.

    Input:
        P: [p, p] table with values in {0, ..., p-1}

    Output:
        P_log: [(p-1), (p-1)] table.

    Procedure:
        1. Restrict rows/cols to nonzero inputs.
        2. Reorder rows/cols by powers of primitive root g.
        3. Convert nonzero output labels z to log_g(z).

    If z=0 occurs on nonzero inputs, it is not in Z_p^*.
    We map it to an extra error symbol by default: p-1.
    For p=97, correct log labels are 0,...,95 and zero_symbol=96.
    """
    if zero_symbol is None:
        zero_symbol = p - 1

    _, order, log = dlog_maps(p)

    P_in = P[np.ix_(order, order)]
    P_out = np.zeros_like(P_in, dtype=np.uint8)

    for i in range(P_in.shape[0]):
        for j in range(P_in.shape[1]):
            z = int(P_in[i, j])
            if z == 0:
                P_out[i, j] = zero_symbol
            else:
                P_out[i, j] = log[z]

    return P_out


# ============================================================
# BDM / entropy computation
# ============================================================

def make_bdm():
    """
    Use fixed 4x4 blocks and ignore boundaries.

    This avoids pybdm normalized=True failures on 97x97 arrays caused
    by recursive boundary shapes without CTM entries.
    """
    try:
        return BDM(
            ndim=2,
            nsymbols=2,
            shape=(4, 4),
            partition=PartitionIgnore,
            warn_if_missing_ctm=False,
            raise_if_zero=False,
        )
    except TypeError:
        # Fallback for older pybdm versions with fewer keyword arguments.
        return BDM(
            ndim=2,
            nsymbols=2,
            shape=(4, 4),
            partition=PartitionIgnore,
        )


def bdm_binary(B):
    """
    Unnormalized BDM for a binary matrix.
    """
    B = np.asarray(B, dtype=int)
    bdm = make_bdm()
    return float(bdm.bdm(B, normalized=False))


def ent_binary(B):
    """
    Unnormalized Shannon entropy baseline from pybdm.
    """
    B = np.asarray(B, dtype=int)
    bdm = make_bdm()

    try:
        return float(bdm.ent(B, normalized=False))
    except TypeError:
        return float(bdm.ent(B))


def bitplane_bdm(P, num_bits=7):
    """
    Compute mean unnormalized BDM and entropy across bit planes.

    P is an integer table.  For p=97, labels fit in 7 bits.
    """
    P = np.asarray(P, dtype=np.uint8)

    vals = []
    ents = []

    for r in range(num_bits):
        B = ((P >> r) & 1).astype(int)
        vals.append(bdm_binary(B))
        ents.append(ent_binary(B))

    return float(np.mean(vals)), float(np.mean(ents))


def bitplane_bdm_per_cell(P, num_bits=7):
    """
    Compute bit-plane BDM / entropy and divide by table size.

    This is the main reported quantity:
        BDM-per-cell = mean_bitplane_BDM / number_of_cells.

    It avoids pybdm normalized=True and makes 97x97 and 96x96
    tables comparable.
    """
    P = np.asarray(P, dtype=np.uint8)
    val, ent = bitplane_bdm(P, num_bits=num_bits)
    return val / P.size, ent / P.size


def correctness_bdm(P, Y):
    """
    BDM and entropy of correctness table:
        C[a,b] = 1[P[a,b] == Y[a,b]]
    """
    C = (np.asarray(P) == np.asarray(Y)).astype(int)
    return bdm_binary(C), ent_binary(C)


def correctness_bdm_per_cell(P, Y):
    """
    Per-cell BDM and entropy of the correctness table.
    """
    val, ent = correctness_bdm(P, Y)
    return val / P.size, ent / P.size


# ============================================================
# Main
# ============================================================

def discover_steps(ckpt_dir):
    steps = []

    for f in os.listdir(ckpt_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.split("_")[-1].split(".")[0])
            steps.append(step)

    return sorted(steps)


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


def compute_seed(args, config, project_root, op_dir, seed, device, aggregate_writer=None):
    p = config.p
    run_dir = resolve_seed_run_dir(
        project_root,
        op_dir,
        seed,
        args.multi_seed,
    )
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_csv = os.path.join(run_dir, "bdm_complexity.csv")

    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return

    os.makedirs(run_dir, exist_ok=True)
    steps = discover_steps(ckpt_dir) if args.steps is None else args.steps
    loader = make_full_loader(p, args.operation, args.batch_size)

    print("=" * 60)
    print("BDM/CTM complexity proxy")
    print(f"operation  = {args.operation}")
    print(f"seed       = {seed}")
    print(f"run dir    = {run_dir}")
    print(f"p          = {p}")
    print(f"num_bits   = {args.num_bits}")
    print(f"device     = {device}")
    print(f"steps      = {len(steps)}")
    print(f"output csv = {out_csv}")
    print("=" * 60)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        f.flush()

        for idx, step in enumerate(steps):
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")

            if not os.path.exists(ckpt_path):
                print(f"[SKIP] step {step}: checkpoint not found")
                continue

            state_dict = load_state_dict(ckpt_path)

            model = GrokkingTransformer(config).to(device)
            model.load_state_dict(state_dict)
            model.eval()

            P, Y = prediction_table(model, loader, p, device)

            # Original coordinate complexity.
            bdm_pred, ent_pred = bitplane_bdm_per_cell(
                P,
                num_bits=args.num_bits,
            )
            bdm_corr, ent_corr = correctness_bdm_per_cell(P, Y)

            # Dlog coordinate complexity for multiplicative operations.
            bdm_pred_dlog = ""
            ent_pred_dlog = ""
            bdm_corr_dlog = ""
            ent_corr_dlog = ""

            if args.operation in ["mul", "div"]:
                P_log = multiplicative_table_dlog(P, p)
                Y_log = multiplicative_table_dlog(Y, p)

                bdm_pred_dlog, ent_pred_dlog = bitplane_bdm_per_cell(
                    P_log,
                    num_bits=args.num_bits,
                )
                bdm_corr_dlog, ent_corr_dlog = correctness_bdm_per_cell(
                    P_log,
                    Y_log,
                )

            row = [
                step,
                f"{bdm_pred:.8f}",
                f"{ent_pred:.8f}",
                f"{bdm_corr:.8f}",
                f"{ent_corr:.8f}",
                f"{bdm_pred_dlog:.8f}" if bdm_pred_dlog != "" else "",
                f"{ent_pred_dlog:.8f}" if ent_pred_dlog != "" else "",
                f"{bdm_corr_dlog:.8f}" if bdm_corr_dlog != "" else "",
                f"{ent_corr_dlog:.8f}" if ent_corr_dlog != "" else "",
            ]
            writer.writerow(row)
            if aggregate_writer is not None:
                aggregate_writer.writerow([seed] + row + [run_dir])
            f.flush()

            msg = (
                f"[{idx + 1}/{len(steps)}] step={step:6d} | "
                f"BDM_pred={bdm_pred:.6f} | "
                f"BDM_correct={bdm_corr:.6f}"
            )

            if args.operation in ["mul", "div"]:
                msg += f" | BDM_pred_dlog={bdm_pred_dlog:.6f}"

            print(msg)

    print(f"\nSaved to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute BDM/CTM algorithmic-complexity proxy."
    )
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
        help="Checkpoint steps. Default: all checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--num-bits",
        type=int,
        default=7,
        help="Number of bit planes. For p=97, use 7.",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Process multiple seed directories. Default seeds: 0 1 2.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds used when --multi-seed is enabled.",
    )
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    seeds = args.seeds if args.multi_seed else [args.seed]

    aggregate_file = None
    aggregate_writer = None
    aggregate_path = os.path.join(
        project_root,
        "data",
        op_dir,
        "bdm_complexity_multiseed.csv",
    )
    if args.multi_seed:
        os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)
        aggregate_file = open(aggregate_path, "w", newline="")
        aggregate_writer = csv.writer(aggregate_file)
        aggregate_writer.writerow(["seed"] + CSV_COLUMNS + ["run_dir"])

    try:
        for seed in seeds:
            compute_seed(
                args,
                config,
                project_root,
                op_dir,
                seed,
                device,
                aggregate_writer,
            )
    finally:
        if aggregate_file is not None:
            aggregate_file.close()
            print(f"\nMulti-seed BDM data saved to: {aggregate_path}")


if __name__ == "__main__":
    main()
