#!/usr/bin/env python3
"""
Plot mode connectivity: linear path vs trained Bezier curve.

Three panels:
    (a) plain train loss along the path (log scale)
    (b) regularized train loss (loss + wd/2 * ||theta||^2) + L2 norm (dashed)
    (c) test accuracy along the path

Reads:
    data/{op}/mode_connectivity.csv                (single seed)
    data/{op}[/seed_{s}]/mode_connectivity.csv     (multi-seed)
Writes:
    results/{op}/mode_connectivity_plot[_multiseed].{pdf,svg}

Usage:
    python utils/mode_connectivity_plot.py --operation add
    python utils/mode_connectivity_plot.py --operation all --multi-seed --seeds 0 1 2
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

DEFAULT_SEEDS = [0, 1, 2]
PATH_STYLES = {"linear": ("tab:gray", "--", "Linear path"),
               "bezier": ("tab:red", "-", "Bezier curve (trained)")}


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


def seed_csvs(project_root, op_dir, seeds, multi_seed):
    data_root = os.path.join(project_root, "data")
    if not multi_seed:
        path = os.path.join(data_root, op_dir, "mode_connectivity.csv")
        if os.path.exists(path):
            yield path
        return
    for seed in seeds:
        candidates = [
            os.path.join(data_root, op_dir, f"seed_{seed}", "mode_connectivity.csv"),
            os.path.join(data_root, f"seed_{seed}", op_dir, "mode_connectivity.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                yield path
                break


def aggregate(all_rows, col, error_mode):
    """Aggregate col over seeds -> {path: (ts, mean, err)}."""
    by_key = defaultdict(list)
    for r in all_rows:
        by_key[(r["path"], float(r["t"]))].append(float(r[col]))

    out = {}
    for path_name in PATH_STYLES:
        ts = sorted({t for (pn, t) in by_key if pn == path_name})
        means, errs = [], []
        for t in ts:
            vals = np.asarray(by_key[(path_name, t)])
            means.append(vals.mean())
            if len(vals) <= 1:
                errs.append(0.0)
            else:
                std = vals.std(ddof=1)
                errs.append(std / np.sqrt(len(vals)) if error_mode == "sem" else std)
        out[path_name] = (np.asarray(ts), np.asarray(means), np.asarray(errs))
    return out


def plot_metric(ax, agg, lw, ylabel, log=False):
    handles = []
    for path_name, (color, ls, label) in PATH_STYLES.items():
        ts, mean, err = agg[path_name]
        if len(ts) == 0:
            continue
        line, = ax.plot(ts, mean, color=color, linestyle=ls, linewidth=lw, label=label)
        ax.fill_between(ts, mean - err, mean + err, color=color, alpha=0.18, linewidth=0)
        handles.append(line)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(r"$t$ (memorization $\to$ generalization)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return handles


def plot_one(op_key, project_root, cfg, multi_seed, seeds, error_mode):
    op_dir, symbol = OPS[op_key]
    all_rows = []
    n_files = 0
    for path in seed_csvs(project_root, op_dir, seeds, multi_seed):
        all_rows.extend(read_rows(path))
        n_files += 1
    if not all_rows:
        print(f"[SKIP] no mode_connectivity.csv found for {op_dir}")
        return

    out_dir = os.path.join(project_root, "results", op_dir)
    os.makedirs(out_dir, exist_ok=True)

    lw = cfg["line"]["linewidth"]
    leg = cfg["legend"]
    figsize = cfg["figure"]["figsize"]
    fig, (ax_loss, ax_reg, ax_acc) = plt.subplots(
        1, 3, figsize=(figsize[0] * 1.8, figsize[1]))

    handles = plot_metric(ax_loss, aggregate(all_rows, "train_loss", error_mode),
                          lw, "Train loss", log=True)
    ax_loss.set_title("Plain train loss")

    plot_metric(ax_reg, aggregate(all_rows, "reg_train_loss", error_mode),
                lw, r"Train loss $+\ \frac{\lambda}{2}\|\theta\|^2$")
    ax_norm = ax_reg.twinx()
    for path_name, (color, _, _) in PATH_STYLES.items():
        ts, mean, _ = aggregate(all_rows, "l2_norm", error_mode)[path_name]
        if len(ts):
            ax_norm.plot(ts, mean, color=color, linestyle=":", linewidth=lw * 0.7, alpha=0.7)
    ax_norm.set_ylabel(r"$\|\theta\|_2$ (dotted)")
    ax_reg.set_title("Regularized potential")

    plot_metric(ax_acc, aggregate(all_rows, "test_acc", error_mode),
                lw, "Test accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.set_title("Generalization along path")

    suffix = f" ({n_files} seeds, {error_mode})" if multi_seed else ""
    fig.suptitle(f"x {symbol} y mod 97 — mode connectivity, "
                 f"memorization vs generalization{suffix}")
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure, ncol=2,
               frameon=leg["frameon"], fontsize=leg["fontsize"])
    fig.subplots_adjust(bottom=0.24, top=0.82, wspace=0.4)

    base = "mode_connectivity_plot_multiseed" if multi_seed else "mode_connectivity_plot"
    for fmt in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"{base}.{fmt}")
        plt.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path}")
    plt.close()

    # barrier summary
    for path_name in PATH_STYLES:
        ts, mean, _ = aggregate(all_rows, "train_loss", error_mode)[path_name]
        if len(ts) == 0:
            continue
        chord = (1 - ts) * mean[0] + ts * mean[-1]
        print(f"    [{path_name:6s}] mean train-loss barrier = {np.max(mean - chord):.6f}")


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
