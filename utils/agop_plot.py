#!/usr/bin/env python3
"""
Plot raw AGOP block heatmaps.

Reads data/{op}/agop/agop_step_{N}.npy and saves each requested block to
results/{op}/agop/.

Usage:
    python utils/agop_plot.py --operation add
    python utils/agop_plot.py --operation all --block xy
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


OPS = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def build_colormap():
    return plt.get_cmap("RdBu_r")


def primitive_root(p):
    """Find a primitive root of prime p."""
    if p == 2:
        return 1
    phi = p - 1
    factors = set()
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"No primitive root found for p={p}")


def dlog_permutation(p):
    """Return token order [g^0, g^1, ..., g^(p-2), 0] for mod-p multiplication."""
    g = primitive_root(p)
    perm = np.zeros(p, dtype=int)
    val = 1
    for k in range(p - 1):
        perm[k] = val
        val = (val * g) % p
    perm[p - 1] = 0
    return perm


def psd_sqrt(mat, eps=1e-10):
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals + eps)) @ vecs.T

def split_blocks(agop, p, use_sqrt=False):
    if use_sqrt:
        agop = psd_sqrt(agop)

    return {
        "xx": agop[:p, :p],
        "xy": agop[:p, p:],
        "yx": agop[p:, :p],
        "yy": agop[p:, p:],
    }


def plot_block(step, block, name, p, out_path, cmap, cfg, perm=None):
    fig, ax = plt.subplots()

    if perm is not None:
        block = block[np.ix_(perm, perm)]

    vmax = np.percentile(np.abs(block), 99)
    if vmax <= 0:
        vmax = np.max(np.abs(block)) + 1e-12

    im = ax.imshow(
        block,
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, p - 1])
    ax.set_yticks([0, p - 1])
    if perm is not None:
        ax.set_xticklabels([str(perm[0]), str(perm[p - 1])])
        ax.set_yticklabels([str(perm[0]), str(perm[p - 1])])
    else:
        ax.set_xticklabels(["0", f"{p - 1}"])
        ax.set_yticklabels(["0", f"{p - 1}"])

    order_tag = " (dlog order)" if perm is not None else ""
    ax.set_xlabel(f"{name[1]} token{order_tag}")
    ax.set_ylabel(f"{name[0]} token{order_tag}")
    ax.set_title(f"G_{name} — Step {step}{order_tag}")

    plt.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] {out_path}")


def process_operation(op_key, project_root, p, block_name, cfg, use_sqrt=False):
    op_dir = OPS[op_key]
    agop_dir = os.path.join(project_root, "data", op_dir, "agop")
    out_dir = os.path.join(project_root, "results", op_dir, "agop")
    perm = dlog_permutation(p) if op_key in ("mul", "div") else None

    if not os.path.isdir(agop_dir):
        print(f"[SKIP] {agop_dir} not found")
        return

    npy_files = sorted(
        [f for f in os.listdir(agop_dir) if f.startswith("agop_step_") and f.endswith(".npy")],
        key=lambda x: int(x.split("_step_")[1].split(".")[0]),
    )
    if not npy_files:
        print(f"[SKIP] No AGOP .npy files in {agop_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    cmap = build_colormap()

    for npy_file in npy_files:
        step = int(npy_file.split("_step_")[1].split(".")[0])
        agop = np.load(os.path.join(agop_dir, npy_file))
        blocks = split_blocks(agop, p, use_sqrt=use_sqrt)
        selected = blocks.keys() if block_name == "all" else [block_name]

        print(f"\nStep {step} | shape={agop.shape}")
        for name in selected:
            out_path = os.path.join(out_dir, f"agop_G{name}_step_{step}.svg")
            plot_block(step, blocks[name], name, p, out_path, cmap, cfg, perm=perm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--block", choices=["xx", "xy", "yx", "yy", "all"], default="all")
    parser.add_argument("--sqrt", action="store_true", help="Plot blocks of sqrt(AGOP)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in ops:
        process_operation(op, project_root, args.p, args.block, cfg, use_sqrt=args.sqrt)


if __name__ == "__main__":
    main()
