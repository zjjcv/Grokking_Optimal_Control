#!/usr/bin/env python3
"""
Plot BDM design-choice sensitivity.

Three panels (encoding / partition / reindexing).  Each panel shows the
prediction-table BDM trajectory of every variant, min-max normalized so the
*shape* is comparable across scales, with the Spearman rank correlation to
the baseline trajectory annotated in the legend.  Test accuracy (mean over
seeds) is drawn as a light background for the grokking transition reference.

Reads:
    data/{op}[/seed_{s}]/bdm_sensitivity.csv
    data/{op}[/seed_{s}]/metric.csv
Writes:
    results/{op}/bdm_sensitivity_plot[_multiseed].{pdf,svg}

Usage:
    python utils/bdm_sensitivity_plot.py --operation add
    python utils/bdm_sensitivity_plot.py --operation all --multi-seed --seeds 0 1 2
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OPS = {
    "add": ("x+y", "+"),
    "sub": ("x-y", "-"),
    "mul": ("x_mul_y", r"\times"),
    "div": ("x_div_y", r"\div"),
}

DEFAULT_SEEDS = [0, 1, 2]

AXES = [
    ("encoding", "Bit-plane encoding"),
    ("partition", "Block partition"),
    ("reindex", "Reindexing"),
]

BASELINE = {"encoding": "binary", "partition": "ignore_4x4", "reindex": "baseline"}

VARIANT_COLORS = {
    "binary": "tab:blue", "gray": "tab:orange", "onehot": "tab:green",
    "ignore_4x4": "tab:blue", "recursive_4x4": "tab:orange",
    "correlated_4x4": "tab:green", "flat_1d_12": "tab:purple",
    "baseline": "tab:blue", "dlog_alt": "tab:orange", "permutation": "tab:red",
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


def seed_files(project_root, op_dir, seeds, multi_seed, filename):
    data_root = os.path.join(project_root, "data")
    if not multi_seed:
        path = os.path.join(data_root, op_dir, filename)
        if os.path.exists(path):
            yield path
        return
    for seed in seeds:
        candidates = [
            os.path.join(data_root, op_dir, f"seed_{seed}", filename),
            os.path.join(data_root, f"seed_{seed}", op_dir, filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                yield path
                break


def mean_curve(vals_by_step):
    steps = sorted(vals_by_step)
    means = np.array([np.mean(vals_by_step[s]) for s in steps])
    stds = np.array([np.std(vals_by_step[s]) for s in steps])
    return np.array(steps), means, stds


def plot_one(op_key, project_root, cfg, multi_seed, seeds, error_mode):
    op_dir, symbol = OPS[op_key]

    # values[axis][variant][step] = list over seeds
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n_files = 0
    for path in seed_files(project_root, op_dir, seeds, multi_seed, "bdm_sensitivity.csv"):
        n_files += 1
        for r in read_rows(path):
            values[r["axis"]][r["variant"]][int(r["step"])].append(
                float(r["bdm_pred_per_cell"]))
    if n_files == 0:
        print(f"[SKIP] no bdm_sensitivity.csv found for {op_dir}")
        return

    # test accuracy background
    acc_by_step = defaultdict(list)
    for path in seed_files(project_root, op_dir, seeds, multi_seed, "metric.csv"):
        for r in read_rows(path):
            acc_by_step[int(float(r["step"]))].append(float(r["test_acc"]))

    out_dir = os.path.join(project_root, "results", op_dir)
    os.makedirs(out_dir, exist_ok=True)

    lw = cfg["line"]["linewidth"]
    figsize = cfg["figure"]["figsize"]
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.8, figsize[1]), sharey=True)

    for ax, (axis_key, axis_title) in zip(axes, AXES):
        # baseline curve for rank correlation
        base_steps, base_mean, _ = mean_curve(values[axis_key][BASELINE[axis_key]])

        if acc_by_step:
            acc_steps, acc_mean, _ = mean_curve(acc_by_step)
            ax.fill_between(acc_steps, 0, acc_mean, color="tab:blue",
                            alpha=0.08, linewidth=0)

        for variant in values[axis_key]:
            steps, mean, std = mean_curve(values[axis_key][variant])
            lo, hi = mean.min(), mean.max()
            span = hi - lo if hi > lo else 1.0
            norm = (mean - lo) / span
            norm_err = std / span

            if variant == BASELINE[axis_key]:
                label = f"{variant} (baseline)"
            else:
                common = [s for s in steps if s in set(base_steps)]
                a = [mean[list(steps).index(s)] for s in common]
                b = [base_mean[list(base_steps).index(s)] for s in common]
                rho = stats.spearmanr(a, b)[0] if len(common) > 2 else np.nan
                label = f"{variant} (ρ={rho:.2f})"

            color = VARIANT_COLORS.get(variant, "tab:gray")
            ax.plot(steps, norm, color=color, linewidth=lw, label=label)
            ax.fill_between(steps, norm - norm_err, norm + norm_err,
                            color=color, alpha=0.15, linewidth=0)

        ax.set_xscale(cfg["axis"]["x_scale"])
        ax.set_xlabel("Step")
        ax.set_title(axis_title)
        ax.grid(True, alpha=cfg["grid"]["alpha"])
        ax.legend(frameon=False, fontsize=10, loc="upper right")

    axes[0].set_ylabel("BDM per cell (min-max normalized)")

    suffix = f" ({n_files} seeds, {error_mode})" if multi_seed else ""
    fig.suptitle(f"x {symbol} y mod 97 — BDM design-choice sensitivity{suffix}"
                 "  (shaded: test accuracy)")
    fig.subplots_adjust(top=0.84, wspace=0.12)

    base = "bdm_sensitivity_plot_multiseed" if multi_seed else "bdm_sensitivity_plot"
    for fmt in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"{base}.{fmt}")
        plt.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--error", choices=["std", "sem"], default="std")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    apply_style(cfg)
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in ops:
        plot_one(op, project_root, cfg, args.multi_seed, args.seeds, args.error)


if __name__ == "__main__":
    main()
