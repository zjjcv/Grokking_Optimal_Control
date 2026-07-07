#!/usr/bin/env python3
"""
2-panel t-SNE plot: Layer 0 hidden | Layer 1 hidden
One figure per checkpoint.

Reads:  data/{op}/pca/tsne_step_{N}.npz
Writes: results/{op}/pca/tsne_2panel_step_{N}.svg

Usage:
    python utils/pca_plot.py --operation add
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

OP_DIR = {
    "add": ("x+y", "+"),
    "sub": ("x-y", "-"),
    "mul": ("x_mul_y", r"\times"),
    "div": ("x_div_y", r"\div"),
}

TOKEN_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
TOKEN_NAMES = [r"$x$", "op", r"$y$"]


def plot_one(step, symbol, data, out_path):
    proj_l0 = data["proj_l0"]
    proj_l1 = data["proj_l1"]
    token_pos = data["token_pos"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, (ax, proj) in enumerate([(axes[0], proj_l0), (axes[1], proj_l1)]):
        for t in range(3):
            mask = token_pos == t
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=TOKEN_COLORS[t], s=4, alpha=0.4,
                       label=TOKEN_NAMES[t], edgecolors="none", rasterized=True)
        ax.legend(loc="best", fontsize=8, markerscale=3)
        ax.set_title(f"Layer {idx} hidden (t-SNE)")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"$x \\ {symbol} \\ y$ mod 97 — Step {step}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[OK] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OP_DIR.keys()), default="add")
    args = parser.parse_args()

    op_dir, symbol = OP_DIR[args.operation]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", op_dir, "pca")
    out_dir = os.path.join(project_root, "results", op_dir, "pca")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        print(f"[SKIP] {data_dir} not found")
        return

    npz_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("tsne_step_") and f.endswith(".npz")],
        key=lambda x: int(x.split("tsne_step_")[1].split(".")[0]),
    )

    for npz_file in npz_files:
        step = int(npz_file.split("tsne_step_")[1].split(".")[0])
        data = dict(np.load(os.path.join(data_dir, npz_file)))
        out_path = os.path.join(out_dir, f"tsne_2panel_step_{step}.svg")
        plot_one(step, symbol, data, out_path)


if __name__ == "__main__":
    main()
