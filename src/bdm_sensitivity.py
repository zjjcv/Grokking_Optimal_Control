#!/usr/bin/env python3
"""
Sensitivity of the BDM/CTM complexity proxy to its design choices.

The baseline pipeline (src/bdm_complexity.py) fixes three arbitrary choices:
binary bit-plane encoding, 4x4 blocks with PartitionIgnore, and (for mul/div)
discrete-log reindexing with the smallest primitive root.  This script varies
each choice along one axis at a time and recomputes the prediction-table BDM
trajectory, so the shape of the conclusion (complexity transition at grokking)
can be checked for invariance.

Axes and variants:
    encoding   binary (baseline) | gray (adjacent values differ by one bit)
               | onehot (97 label-indicator planes; no positional bit code)
    partition  ignore_4x4 (baseline) | recursive_4x4 | correlated_4x4
               (sliding window) | flat_1d_12 (row-major 1D BDM, 12-bit blocks)
    reindex    baseline (identity for add/sub, dlog smallest primitive root
               for mul/div) | dlog_alt (mul/div only: different primitive
               root) | permutation (random row/col/label permutation;
               negative control - should destroy structure)

Outputs:
    data/{op}[/seed_{s}]/bdm_sensitivity.csv
    data/{op}/bdm_sensitivity_multiseed.csv   (multi-seed aggregate)

Usage:
    python src/bdm_sensitivity.py --operation add
    python src/bdm_sensitivity.py --operation mul --multi-seed --seeds 0 1 2
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, GrokkingTransformer
from bdm_complexity import (
    OP_DIR,
    DEFAULT_SEEDS,
    make_full_loader,
    prediction_table,
    primitive_root_mod_p,
    discover_steps,
    load_state_dict,
    resolve_seed_run_dir,
)

from pybdm import BDM
from pybdm.partitions import PartitionIgnore, PartitionRecursive, PartitionCorrelated


NUM_BITS = 7
PERM_SEED = 12345


# ---------- BDM calculators per partition variant ----------

def bdm_calculators():
    calcs = {
        "ignore_4x4": BDM(ndim=2, nsymbols=2, shape=(4, 4), partition=PartitionIgnore,
                          warn_if_missing_ctm=False, raise_if_zero=False),
        "recursive_4x4": BDM(ndim=2, nsymbols=2, shape=(4, 4), partition=PartitionRecursive,
                             warn_if_missing_ctm=False, raise_if_zero=False),
        "correlated_4x4": BDM(ndim=2, nsymbols=2, shape=(4, 4), partition=PartitionCorrelated,
                              warn_if_missing_ctm=False, raise_if_zero=False),
        "flat_1d_12": BDM(ndim=1, nsymbols=2, shape=(12,), partition=PartitionIgnore,
                          warn_if_missing_ctm=False, raise_if_zero=False),
    }
    return calcs


def bdm_of_binary(calc, B, flat):
    B = np.asarray(B, dtype=int)
    if flat:
        B = B.reshape(-1)
    return float(calc.bdm(B, normalized=False))


# ---------- encodings ----------

def planes_binary(P, num_bits=NUM_BITS):
    return [((P >> r) & 1).astype(int) for r in range(num_bits)]


def planes_gray(P, num_bits=NUM_BITS):
    G = P ^ (P >> 1)
    return [((G >> r) & 1).astype(int) for r in range(num_bits)]


def planes_onehot(P, p):
    return [(P == c).astype(int) for c in range(p)]


def encode(P, encoding, p):
    if encoding == "binary":
        return planes_binary(P)
    if encoding == "gray":
        return planes_gray(P)
    if encoding == "onehot":
        return planes_onehot(P, p)
    raise ValueError(encoding)


# ---------- reindexings ----------

def dlog_reindex(P, p, g):
    """Restrict to nonzero inputs ordered by powers of g; relabel outputs by log_g."""
    order, log = [], {}
    x = 1
    for k in range(p - 1):
        order.append(x)
        log[x] = k
        x = (x * g) % p

    P_in = P[np.ix_(order, order)]
    P_out = np.zeros_like(P_in, dtype=np.int64)
    zero_symbol = p - 1
    for i in range(P_in.shape[0]):
        for j in range(P_in.shape[1]):
            z = int(P_in[i, j])
            P_out[i, j] = zero_symbol if z == 0 else log[z]
    return P_out


def second_primitive_root(p):
    g1 = primitive_root_mod_p(p)
    for g in range(g1 + 1, p):
        # check primitivity by brute force on the factors of p-1
        factors = []
        x, q = p - 1, 2
        while q * q <= x:
            if x % q == 0:
                factors.append(q)
                while x % q == 0:
                    x //= q
            q += 1
        if x > 1:
            factors.append(x)
        if all(pow(g, (p - 1) // f, p) != 1 for f in factors):
            return g
    raise RuntimeError("no second primitive root")


def permutation_reindex(P, p, rng, relabel=True):
    """Random row/col reorder + optional label relabeling (negative control).

    relabel=False for binary tables (e.g. correctness), where only the
    row/column order is meaningful.
    """
    sr = rng.permutation(p)
    sc = rng.permutation(p)
    out = P[np.ix_(sr, sc)]
    if relabel:
        sl = rng.permutation(p)
        out = sl[out]
    return out


def reindex_table(P, variant, operation, p, rng):
    if variant == "baseline":
        if operation in ("mul", "div"):
            return dlog_reindex(P, p, primitive_root_mod_p(p))
        return P.astype(np.int64)
    if variant == "dlog_alt":
        return dlog_reindex(P, p, second_primitive_root(p))
    if variant == "permutation":
        return permutation_reindex(P.astype(np.int64), p, rng)
    raise ValueError(variant)


# ---------- per-table metric ----------

def bdm_per_cell(planes, calc, flat):
    vals = [bdm_of_binary(calc, B, flat) / B.size for B in planes]
    return float(np.mean(vals))


def compute_variants(P, C, operation, p, calcs):
    """All one-axis-at-a-time variants for one checkpoint.

    P: prediction table [p, p]; C: binary correctness table [p, p].
    Returns list of (axis, variant, bdm_pred, bdm_correct or "").
    """
    rng = np.random.default_rng(PERM_SEED)
    base_calc = calcs["ignore_4x4"]
    P_base = reindex_table(P, "baseline", operation, p, rng)
    rows = []

    # axis 1: encoding (partition + reindex fixed at baseline)
    for encoding in ("binary", "gray", "onehot"):
        val = bdm_per_cell(encode(P_base, encoding, p), base_calc, flat=False)
        # correctness table is already binary: encoding axis does not apply
        rows.append(("encoding", encoding, val, ""))

    # axis 2: partition (encoding binary, reindex baseline)
    for part in ("ignore_4x4", "recursive_4x4", "correlated_4x4", "flat_1d_12"):
        calc = calcs[part]
        flat = part == "flat_1d_12"
        val = bdm_per_cell(planes_binary(P_base), calc, flat)
        val_c = bdm_of_binary(calc, C, flat) / C.size
        rows.append(("partition", part, val, val_c))

    # axis 3: reindexing (encoding binary, partition baseline)
    variants = ["baseline", "permutation"]
    if operation in ("mul", "div"):
        variants.insert(1, "dlog_alt")
    for variant in variants:
        rng_v = np.random.default_rng(PERM_SEED)
        P_r = reindex_table(P, variant, operation, p, rng_v)
        val = bdm_per_cell(planes_binary(P_r), base_calc, flat=False)
        if variant == "permutation":
            rng_c = np.random.default_rng(PERM_SEED)
            C_r = permutation_reindex(C.astype(np.int64), p, rng_c, relabel=False)
        else:
            C_r = C
        val_c = bdm_of_binary(base_calc, C_r, flat=False) / C_r.size
        rows.append(("reindex", variant, val, val_c))

    return rows


# ---------- driver ----------

def compute_seed(args, config, project_root, op_dir, seed, device, aggregate_writer=None):
    p = config.p
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_csv = os.path.join(run_dir, "bdm_sensitivity.csv")
    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return

    steps = args.steps or [s for s in discover_steps(ckpt_dir) if s % args.step_interval == 0]
    loader = make_full_loader(p, args.operation, args.batch_size)
    calcs = bdm_calculators()

    print(f"seed={seed}  run_dir={run_dir}  steps={len(steps)}")
    fields = ["step", "seed", "axis", "variant", "bdm_pred_per_cell", "bdm_correct_per_cell"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        f.flush()

        for idx, step in enumerate(steps):
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            if not os.path.exists(ckpt_path):
                continue
            model = GrokkingTransformer(config).to(device)
            model.load_state_dict(load_state_dict(ckpt_path))
            model.eval()

            P, Y = prediction_table(model, loader, p, device)
            C = (P == Y).astype(int)

            for axis, variant, val, val_c in compute_variants(P.astype(np.int64), C,
                                                              args.operation, p, calcs):
                row = [step, seed, axis, variant, f"{val:.8f}",
                       f"{val_c:.8f}" if val_c != "" else ""]
                writer.writerow(row)
                if aggregate_writer is not None:
                    aggregate_writer.writerow(row + [run_dir])
            f.flush()

            if (idx + 1) % 10 == 0 or idx == len(steps) - 1:
                print(f"  [{idx + 1}/{len(steps)}] seed={seed} step={step}")

    print(f"Saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="BDM design-choice sensitivity analysis")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--step-interval", type=int, default=1000,
                        help="Used when --steps is omitted: every N-th checkpoint step.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    seeds = args.seeds if args.multi_seed else [args.seed]

    aggregate_file, aggregate_writer = None, None
    if args.multi_seed:
        aggregate_path = os.path.join(project_root, "data", op_dir, "bdm_sensitivity_multiseed.csv")
        aggregate_file = open(aggregate_path, "w", newline="")
        aggregate_writer = csv.writer(aggregate_file)
        aggregate_writer.writerow(["step", "seed", "axis", "variant",
                                   "bdm_pred_per_cell", "bdm_correct_per_cell", "run_dir"])

    try:
        for seed in seeds:
            compute_seed(args, config, project_root, op_dir, seed, device, aggregate_writer)
    finally:
        if aggregate_file is not None:
            aggregate_file.close()
            print("\nMulti-seed aggregate saved.")


if __name__ == "__main__":
    main()
