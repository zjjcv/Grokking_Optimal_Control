#!/usr/bin/env python3
"""
Kolmogorov complexity (BDM) for model weights, AGOP matrices, and NFM matrices.
Also computes circulant/anti-circulant alignment: R_circ, R_anti.

Outputs:
    data/{op}/kc.csv        — per-parameter weight KC
    data/{op}/kc_agop.csv   — AGOP KC + R_circ/R_anti (full + sub-blocks)
    data/{op}/kc_nfm.csv    — NFM KC + R_circ/R_anti per component

Usage:
    python src/kc.py --operation add
    python src/kc.py --operation mul --steps 1000 10000 90000
"""

import os
import re
import csv
import argparse

import numpy as np
import torch
from pybdm import BDM

from train import Config

OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}


# ==================== Binarize & KC ====================

def binarize(arr):
    threshold = arr.mean()
    return (arr > threshold).astype(np.int8)


def compute_kc(bdm2d, bdm1d, state_dict):
    """Compute KC (BDM) for every parameter in state_dict."""
    results = {}
    for name, tensor in sorted(state_dict.items()):
        arr = tensor.detach().cpu().numpy()
        try:
            if arr.ndim == 1:
                results[name] = bdm1d.bdm(binarize(arr))
            elif arr.ndim == 2:
                results[name] = bdm2d.bdm(binarize(arr))
            else:
                flat = arr.reshape(-1, arr.shape[-1])
                results[name] = bdm2d.bdm(binarize(flat))
        except ValueError:
            results[name] = float("nan")
    return results


# ==================== Circulant / Anti-circulant ====================

def circulant_project(A):
    """Project A onto wrapped-diagonal (circulant) subspace.
    C[i, (i+d) mod p] = mean_i A[i, (i+d) mod p] for each offset d.
    """
    p = A.shape[0]
    C = np.zeros_like(A)
    for d in range(p):
        vals = np.array([A[i, (i + d) % p] for i in range(p)])
        m = vals.mean()
        for i in range(p):
            C[i, (i + d) % p] = m
    return C


def anti_circulant_project(A):
    """Project A onto wrapped anti-diagonal subspace.
    C[i, (s-i) mod p] = mean_i A[i, (s-i) mod p] for each sum s.
    """
    p = A.shape[0]
    C = np.zeros_like(A)
    for s in range(p):
        vals = np.array([A[i, (s - i) % p] for i in range(p)])
        m = vals.mean()
        for i in range(p):
            C[i, (s - i) % p] = m
    return C


def circulant_alignment(A):
    """R_circ(A) = ||P_circ(A)||_F^2 / ||A||_F^2"""
    P = circulant_project(A)
    return float(np.sum(P ** 2) / (np.sum(A ** 2) + 1e-30))


def anti_circulant_alignment(A):
    """R_anti(A) = ||P_anti(A)||_F^2 / ||A||_F^2"""
    P = anti_circulant_project(A)
    return float(np.sum(P ** 2) / (np.sum(A ** 2) + 1e-30))


# ==================== AGOP analysis ====================

AGOP_BLOCKS = ["full", "Gxx", "Gxy", "Gyy"]


def compute_agop_metrics(bdm2d, agop_path, p):
    """KC + R_circ/R_anti for AGOP full matrix and sub-blocks."""
    agop = np.load(agop_path)
    G_xx = agop[:p, :p]
    G_yy = agop[p:, p:]
    G_xy = agop[:p, p:]

    results = {}
    for name, mat in [("full", agop), ("Gxx", G_xx), ("Gxy", G_xy), ("Gyy", G_yy)]:
        try:
            results[f"kc_{name}"] = float(bdm2d.bdm(binarize(mat)))
        except ValueError:
            results[f"kc_{name}"] = float("nan")
        results[f"R_circ_{name}"] = circulant_alignment(mat)
        results[f"R_anti_{name}"] = anti_circulant_alignment(mat)
    return results


# ==================== NFM analysis ====================

def discover_nfm_components(nfm_dir):
    """Discover all unique component names from NFM .npy files."""
    comps = set()
    comp_re = re.compile(r"nfm_(.+)_step_\d+\.npy")
    for f in os.listdir(nfm_dir):
        m = comp_re.match(f)
        if m:
            comps.add(m.group(1))
    return sorted(comps)


def compute_nfm_metrics(bdm2d, nfm_dir, step):
    """KC + R_circ/R_anti for all NFM components at a given step."""
    results = {}
    step_re = re.compile(rf"nfm_(.+)_step_{step}\.npy")
    for f in sorted(os.listdir(nfm_dir)):
        m = step_re.match(f)
        if not m:
            continue
        comp = m.group(1)
        mat = np.load(os.path.join(nfm_dir, f))
        try:
            results[f"kc_{comp}"] = float(bdm2d.bdm(binarize(mat)))
        except ValueError:
            results[f"kc_{comp}"] = float("nan")
        results[f"R_circ_{comp}"] = circulant_alignment(mat)
        results[f"R_anti_{comp}"] = anti_circulant_alignment(mat)
    return results


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Compute KC for weights, AGOP, and NFM")
    parser.add_argument("--operation", default="add", choices=["add", "sub", "mul", "div"])
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Checkpoint steps. Default: 9 representative points.")
    args = parser.parse_args()

    config = Config(args.operation)
    p = config.p

    op_dir = OP_DIR[args.operation]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, "data", op_dir, "checkpoints")
    agop_dir = os.path.join(project_root, "data", op_dir, "agop")
    nfm_dir = os.path.join(project_root, "data", op_dir, "nfm")

    if args.steps is None:
        args.steps = [100, 500, 1000, 3000, 5000, 10000, 30000, 50000, 90000]

    bdm2d = BDM(ndim=2)
    bdm1d = BDM(ndim=1)

    # ==================== 1. Weight KC ====================
    param_names = None
    for step in args.steps:
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
            param_names = sorted(sd.keys())
            break

    if param_names is not None:
        out_csv = os.path.join(project_root, "data", op_dir, "kc.csv")
        print("=" * 60)
        print("Weight KC (BDM) computation")
        print(f"operation  = {args.operation}")
        print(f"params     = {len(param_names)}")
        print(f"steps      = {args.steps}")
        print(f"output     = {out_csv}")
        print("=" * 60)

        header = ["step"] + param_names
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            f.flush()
            for idx, step in enumerate(args.steps):
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
                if not os.path.exists(ckpt_path):
                    print(f"  [SKIP] step {step}: checkpoint not found")
                    continue
                state = torch.load(ckpt_path, map_location="cpu")
                state_dict = (state["model_state_dict"]
                              if isinstance(state, dict) and "model_state_dict" in state
                              else state)
                kc = compute_kc(bdm2d, bdm1d, state_dict)
                row = [step] + [f"{kc[n]:.6f}" for n in param_names]
                writer.writerow(row)
                f.flush()
                print(f"  [{idx + 1}/{len(args.steps)}] Step {step} done")
        print(f"Weight KC saved to: {out_csv}\n")
    else:
        print("[SKIP] No weight checkpoints found.\n")

    # ==================== 2. AGOP KC + alignment ====================
    if os.path.isdir(agop_dir):
        agop_files = {
            int(f.split("_step_")[1].split(".")[0]): f
            for f in os.listdir(agop_dir)
            if f.startswith("agop_step_") and f.endswith(".npy")
        }
        steps_agop = sorted(s for s in args.steps if s in agop_files)

        if steps_agop:
            out_csv = os.path.join(project_root, "data", op_dir, "kc_agop.csv")
            agop_cols = []
            for blk in AGOP_BLOCKS:
                agop_cols += [f"kc_{blk}", f"R_circ_{blk}", f"R_anti_{blk}"]

            print("=" * 60)
            print("AGOP KC + circulant alignment")
            print(f"operation  = {args.operation}")
            print(f"steps      = {steps_agop}")
            print(f"output     = {out_csv}")
            print("=" * 60)

            with open(out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step"] + agop_cols)
                f.flush()
                for idx, step in enumerate(steps_agop):
                    agop_path = os.path.join(agop_dir, agop_files[step])
                    results = compute_agop_metrics(bdm2d, agop_path, p)
                    row = [step] + [f"{results[c]:.6f}" for c in agop_cols]
                    writer.writerow(row)
                    f.flush()
                    print(f"  [{idx + 1}/{len(steps_agop)}] Step {step} done")
            print(f"AGOP KC saved to: {out_csv}\n")
        else:
            print("[SKIP] No AGOP data for requested steps.\n")
    else:
        print(f"[SKIP] {agop_dir} not found.\n")

    # ==================== 3. NFM KC + alignment ====================
    if os.path.isdir(nfm_dir):
        all_components = discover_nfm_components(nfm_dir)
        # Discover available steps
        step_re = re.compile(r"nfm_.+_step_(\d+)\.npy")
        available_steps = sorted(set(
            int(step_re.search(f).group(1))
            for f in os.listdir(nfm_dir) if step_re.search(f)
        ))
        steps_nfm = sorted(s for s in args.steps if s in available_steps)

        if steps_nfm and all_components:
            out_csv = os.path.join(project_root, "data", op_dir, "kc_nfm.csv")
            nfm_cols = []
            for comp in all_components:
                nfm_cols += [f"kc_{comp}", f"R_circ_{comp}", f"R_anti_{comp}"]

            print("=" * 60)
            print("NFM KC + circulant alignment")
            print(f"operation   = {args.operation}")
            print(f"components  = {len(all_components)}")
            print(f"steps       = {steps_nfm}")
            print(f"output      = {out_csv}")
            print("=" * 60)

            with open(out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step"] + nfm_cols)
                f.flush()
                for idx, step in enumerate(steps_nfm):
                    results = compute_nfm_metrics(bdm2d, nfm_dir, step)
                    row = [step]
                    for comp in all_components:
                        for metric in [f"kc_{comp}", f"R_circ_{comp}", f"R_anti_{comp}"]:
                            row.append(f"{results.get(metric, float('nan')):.6f}")
                    writer.writerow(row)
                    f.flush()
                    print(f"  [{idx + 1}/{len(steps_nfm)}] Step {step} done")
            print(f"NFM KC saved to: {out_csv}\n")
        else:
            print("[SKIP] No NFM data for requested steps.\n")
    else:
        print(f"[SKIP] {nfm_dir} not found.\n")

    print("All done.")


if __name__ == "__main__":
    main()
