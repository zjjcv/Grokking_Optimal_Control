#!/usr/bin/env python3
"""
Plot Causal Mediation Analysis (mean ablation) results across training steps.

Reads:  data/{op}/cma/cma.csv
        data/{op}/metric.csv
Writes: results/{op}/cma/cma_evolution.svg    — line plot of test drop per head vs step
        results/{op}/cma/cma_bar_step_{N}.svg  — bar chart per step

Usage:
    python utils/cma_plot.py --operation add
    python utils/cma_plot.py --operation all
"""

import os
import json
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OPS = {
    "add": ("x+y", "+"),
    "sub": ("x-y", "-"),
    "mul": ("x_mul_y", r"\times"),
    "div": ("x_div_y", r"\div"),
}

HEAD_COLORS = {
    "l0_h0": "#4C72B0", "l0_h1": "#55A868", "l0_h2": "#C44E52", "l0_h3": "#8172B2",
    "l1_h0": "#CCB974", "l1_h1": "#64B5CD", "l1_h2": "#E57C77", "l1_h3": "#8C8C8C",
}
HEAD_MARKERS = {
    "l0_h0": "o", "l0_h1": "s", "l0_h2": "^", "l0_h3": "D",
    "l1_h0": "o", "l1_h1": "s", "l1_h2": "^", "l1_h3": "D",
}


def load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


def read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def read_metric_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        data = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                data[c].append(float(row[c]))
    return data


def plot_evolution(op_key, project_root, cfg):
    """Line plot: test accuracy drop per head across training steps."""
    op_dir, symbol = OPS[op_key]
    csv_path = os.path.join(project_root, "data", op_dir, "cma", "cma.csv")
    metric_path = os.path.join(project_root, "data", op_dir, "metric.csv")
    out_dir = os.path.join(project_root, "results", op_dir, "cma")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found")
        return

    rows = read_csv(csv_path)

    # Organize: head -> {step: (train_drop, test_drop)}
    head_data = defaultdict(dict)
    steps_set = set()
    for r in rows:
        s = int(r["step"])
        head_data[r["head"]][s] = (float(r["train_acc_drop"]), float(r["test_acc_drop"]))
        steps_set.add(s)
    steps = sorted(steps_set)
    heads = sorted(head_data.keys())

    lw = cfg["line"]["linewidth"]
    metric = read_metric_csv(metric_path) if os.path.exists(metric_path) else None

    for split, ylabel, acc_col, suffix in [
        ("test", "Test Accuracy Drop", "test_acc", "cma_evolution_test.svg"),
        ("train", "Train Accuracy Drop", "train_acc", "cma_evolution_train.svg"),
    ]:
        fig, ax_drop = plt.subplots(figsize=(11, 5.5))

        for head in heads:
            d = [head_data[head].get(s, (0, 0))[0 if split == "train" else 1] for s in steps]
            ax_drop.plot(steps, d, color=HEAD_COLORS[head], marker=HEAD_MARKERS[head],
                         linewidth=lw, markersize=5, label=head, alpha=0.85)

        ax_drop.set_xscale(cfg["axis"]["x_scale"])
        ax_drop.set_xlabel("Step")
        ax_drop.set_ylabel(ylabel)
        ax_drop.set_ylim(-0.05, 1.05)
        ax_drop.grid(True, alpha=cfg["grid"]["alpha"])

        # Overlay accuracy on right axis
        if metric is not None:
            ax_acc = ax_drop.twinx()
            c_acc = cfg["color"]["accuracy"]
            ax_acc.plot(metric["step"], metric[acc_col], color=c_acc,
                        linestyle="--", linewidth=lw * 0.7, alpha=0.4)
            ax_acc.set_ylabel(acc_col.replace("_", " ").title(), color=c_acc)
            ax_acc.tick_params(axis="y", labelcolor=c_acc)
            ax_acc.set_ylim(-0.05, 1.05)

        ax_drop.set_title(f"$x \\ {symbol} \\ y$ mod 97 — CMA ({split.title()})")

        l0_handles = [Line2D([0], [0], color=HEAD_COLORS[h], marker=HEAD_MARKERS[h],
                             linewidth=1.5, label=h) for h in heads if h.startswith("l0")]
        l1_handles = [Line2D([0], [0], color=HEAD_COLORS[h], marker=HEAD_MARKERS[h],
                             linewidth=1.5, label=h) for h in heads if h.startswith("l1")]
        leg = cfg["legend"]
        ax_drop.legend(handles=l0_handles + l1_handles,
                       labels=[h.get_label() for h in l0_handles + l1_handles],
                       loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
                       bbox_transform=fig.transFigure, ncol=4,
                       frameon=leg["frameon"], fontsize=9)
        fig.subplots_adjust(bottom=0.18)

        svg_path = os.path.join(out_dir, suffix)
        plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"[OK] {svg_path}")


def plot_bars(op_key, project_root, cfg):
    """Per-step bar chart (same layout as old single-step plot)."""
    op_dir, symbol = OPS[op_key]
    csv_path = os.path.join(project_root, "data", op_dir, "cma", "cma.csv")
    out_dir = os.path.join(project_root, "results", op_dir, "cma")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        return

    rows = read_csv(csv_path)
    steps = sorted(set(int(r["step"]) for r in rows))
    heads_order = [f"l{l}_h{h}" for l in range(2) for h in range(4)]
    layers_map = {f"l{l}_h{h}": l for l in range(2) for h in range(4)}

    c_l0, c_l1 = "#4C72B0", "#DD8452"
    width = 0.35

    for step in steps:
        step_rows = [r for r in rows if int(r["step"]) == step]
        row_map = {r["head"]: r for r in step_rows}
        ordered = [row_map[h] for h in heads_order if h in row_map]
        if not ordered:
            continue

        n = len(ordered)
        x = np.arange(n)
        train_drops = [float(r["train_acc_drop"]) for r in ordered]
        test_drops = [float(r["test_acc_drop"]) for r in ordered]
        head_labels = [r["head"] for r in ordered]
        layer_labels = [layers_map[h] for h in head_labels]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i in range(n):
            c = c_l0 if layer_labels[i] == 0 else c_l1
            ax.bar(x[i] - width / 2, train_drops[i], width, color=c, alpha=0.5,
                   edgecolor="black", linewidth=0.5)
            ax.bar(x[i] + width / 2, test_drops[i], width, color=c, alpha=0.9,
                   edgecolor="black", linewidth=0.5, hatch="//")

        sep = sum(1 for l in layer_labels if l == 0) - 0.5
        ymax = max(max(train_drops), max(test_drops), 0.01) * 1.2
        ax.axvline(x=sep, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(sep / 2 - 0.5, ymax, "L0", ha="center", fontsize=9, color="gray")
        ax.text(sep + (n - sum(1 for l in layer_labels if l == 0)) / 2 - 0.5, ymax,
                "L1", ha="center", fontsize=9, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(head_labels, fontsize=9)
        ax.set_ylabel("Accuracy Drop")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=cfg["grid"]["alpha"], axis="y")
        ax.set_title(f"$x \\ {symbol} \\ y$ mod 97 — CMA Step {step}")
        plt.tight_layout()

        svg_path = os.path.join(out_dir, f"cma_bar_step_{step}.svg")
        plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"[OK] {svg_path}")


def plot_one(op_key, project_root, cfg):
    plot_evolution(op_key, project_root, cfg)
    plot_bars(op_key, project_root, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]

    for op in ops:
        plot_one(op, project_root, cfg)


if __name__ == "__main__":
    main()
