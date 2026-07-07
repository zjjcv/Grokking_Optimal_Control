#!/usr/bin/env python3
"""
Plot activation patching (causal tracing) results.

Two annotated heatmaps (heads x checkpoints):
    left  - denoising recovery  (clean activation patched into corrupted run)
    right - noising damage      (corrupted activation patched into clean run)

Reads:
    data/{op}/cma/patching.csv
Writes:
    results/{op}/cma/activation_patching_plot.{pdf,svg}

Usage:
    python utils/activation_patching_plot.py --operation add
    python utils/activation_patching_plot.py --operation all
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

OPS = {
    "add": ("x+y", "+"),
    "sub": ("x-y", "-"),
    "mul": ("x_mul_y", r"\times"),
    "div": ("x_div_y", r"\div"),
}


def load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


def apply_style(cfg):
    style = cfg.get("style", {})
    weight = style.get("font_weight", "bold")
    plt.rcParams.update({
        "axes.linewidth": style.get("spine_linewidth", 1.8),
        "xtick.major.width": style.get("tick_width", 1.6),
        "ytick.major.width": style.get("tick_width", 1.6),
        "xtick.minor.width": style.get("tick_width", 1.6) * 0.6,
        "ytick.minor.width": style.get("tick_width", 1.6) * 0.6,
        "font.size": style.get("font_size", 12),
        "font.weight": weight,
        "axes.labelweight": weight,
        "axes.titleweight": weight,
        "figure.titleweight": weight,
    })


def read_rows(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def draw_heatmap(ax, matrix, steps, heads, title, cbar_label, fig):
    vmax = max(1.0, np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heads)))
    ax.set_yticklabels(heads)
    ax.set_xlabel("Step")
    ax.set_ylabel("Attention head")
    ax.set_title(title)

    threshold = 0.55 * vmax
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > threshold else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(plt.rcParams["axes.linewidth"] * 0.8)


def plot_one(op_key, project_root, cfg):
    op_dir, symbol = OPS[op_key]
    csv_path = os.path.join(project_root, "data", op_dir, "cma", "patching.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found")
        return
    rows = read_rows(csv_path)

    out_dir = os.path.join(project_root, "results", op_dir, "cma")
    os.makedirs(out_dir, exist_ok=True)

    recovery = defaultdict(dict)  # head -> step -> value
    damage = defaultdict(dict)
    for r in rows:
        step = int(r["step"])
        recovery[r["head"]][step] = float(r["recovery"])
        damage[r["head"]][step] = float(r["damage"])

    heads = sorted(recovery)
    steps = sorted({int(r["step"]) for r in rows})

    mat_rec = np.array([[recovery[h].get(s, np.nan) for s in steps] for h in heads])
    mat_dam = np.array([[damage[h].get(s, np.nan) for s in steps] for h in heads])

    figsize = cfg["figure"]["figsize"]
    fig, (ax_rec, ax_dam) = plt.subplots(
        1, 2, figsize=(figsize[0] * 1.5, figsize[1] * 1.1))

    draw_heatmap(ax_rec, mat_rec, steps, heads,
                 "Denoising: clean → corrupted run", "Logit-diff recovery", fig)
    draw_heatmap(ax_dam, mat_dam, steps, heads,
                 "Noising: corrupted → clean run", "Logit-diff damage", fig)

    fig.suptitle(f"x {symbol} y mod 97 — activation patching (causal tracing)")
    fig.subplots_adjust(wspace=0.35, top=0.85)

    for fmt in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"activation_patching_plot.{fmt}")
        plt.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    apply_style(cfg)
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in ops:
        plot_one(op, project_root, cfg)


if __name__ == "__main__":
    main()
