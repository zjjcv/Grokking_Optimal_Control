#!/usr/bin/env python3
"""
Plot Gaussian perturbation and random-ablation robustness.

Single-seed mode reads:
    data/{op}/perturb/perturb.csv
or legacy:
    data/{op}/perturb.csv

Multi-seed mode reads:
    data/{op}/perturb/perturb_multiseed.csv
or per-seed:
    data/{op}/seed_{seed}/perturb/perturb.csv
    data/seed_{seed}/{op}/perturb/perturb.csv
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


def load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


def read_rows(path):
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = np.nan
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def seed_paths(project_root, op_dir, seeds):
    data_root = os.path.join(project_root, "data")
    for seed in seeds:
        candidates = [
            os.path.join(data_root, op_dir, f"seed_{seed}", "perturb", "perturb.csv"),
            os.path.join(data_root, f"seed_{seed}", op_dir, "perturb", "perturb.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                yield path
                break


def load_rows(project_root, op_dir, multi_seed, seeds):
    data_dir = os.path.join(project_root, "data", op_dir)
    if multi_seed:
        aggregate = os.path.join(data_dir, "perturb", "perturb_multiseed.csv")
        if os.path.exists(aggregate):
            return read_rows(aggregate)
        rows = []
        for path in seed_paths(project_root, op_dir, seeds):
            rows.extend(read_rows(path))
        return rows

    preferred = os.path.join(data_dir, "perturb", "perturb.csv")
    legacy = os.path.join(data_dir, "perturb.csv")
    if os.path.exists(preferred):
        return read_rows(preferred)
    if os.path.exists(legacy):
        return read_rows(legacy)
    return []


def save_all(fig, base):
    for fmt in ("pdf", "svg"):
        path = f"{base}.{fmt}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {path}")


def aggregate(rows, col, error_mode):
    by_step = defaultdict(list)
    for row in rows:
        value = row.get(col, np.nan)
        if not np.isnan(value):
            by_step[row["step"]].append(value)
    steps, means, errs = [], [], []
    for step in sorted(by_step):
        vals = np.asarray(by_step[step], dtype=float)
        steps.append(step)
        means.append(float(vals.mean()))
        if len(vals) <= 1:
            errs.append(0.0)
        else:
            std = float(vals.std(ddof=1))
            errs.append(std / np.sqrt(len(vals)) if error_mode == "sem" else std)
    return np.asarray(steps), np.asarray(means), np.asarray(errs)


def plot_line(ax, rows, col, label, color, linestyle, lw, error_mode=None, band=False):
    steps, mean, err = aggregate(rows, col, error_mode or "std")
    if len(steps) == 0:
        return None
    line, = ax.plot(steps, mean, color=color, linestyle=linestyle, linewidth=lw, label=label)
    if band:
        ax.fill_between(steps, mean - err, mean + err, color=color, alpha=0.16, linewidth=0)
    return line


def plot_accuracy(symbol, rows, out_dir, cfg, multi_seed, error_mode):
    lw = cfg["line"]["linewidth"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = {"orig": "#2F6FBB", "noise": "#D55E00", "rand": "#7A5195"}

    plot_line(ax, rows, "train_acc_orig", "Train original", colors["orig"], "--", lw * 0.75, error_mode)
    plot_line(ax, rows, "test_acc_orig", "Test original", colors["orig"], "-", lw, error_mode,
              band=multi_seed)
    plot_line(ax, rows, "train_acc_perturb", "Train Gaussian", colors["noise"], "--", lw * 0.75, error_mode)
    plot_line(ax, rows, "test_acc_perturb", "Test Gaussian", colors["noise"], "-", lw, error_mode,
              band=True)
    plot_line(ax, rows, "train_acc_random_ablation", "Train random ablation", colors["rand"], "--",
              lw * 0.75, error_mode)
    plot_line(ax, rows, "test_acc_random_ablation", "Test random ablation", colors["rand"], "-",
              lw, error_mode, band=True)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_xscale(cfg["axis"]["x_scale"])
    ax.set_ylim(cfg["axis"]["acc_ylim"])
    ax.grid(True, alpha=cfg["grid"]["alpha"])
    suffix = f" ({error_mode} band)" if multi_seed else ""
    ax.set_title(f"x {symbol} y mod 97 - Perturbation and Random-Ablation Robustness{suffix}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(bottom=0.3)
    save_all(fig, os.path.join(out_dir, "perturb_accuracy_multiseed" if multi_seed else "perturb_accuracy"))
    plt.close(fig)


def plot_drop(symbol, rows, out_dir, cfg, multi_seed, error_mode):
    lw = cfg["line"]["linewidth"]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    colors = {"noise": "#D55E00", "rand": "#7A5195"}
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    plot_line(ax, rows, "train_drop_perturb", "Train Gaussian drop", colors["noise"], "--",
              lw * 0.75, error_mode)
    plot_line(ax, rows, "test_drop_perturb", "Test Gaussian drop", colors["noise"], "-",
              lw, error_mode, band=True)
    plot_line(ax, rows, "train_drop_random_ablation", "Train random-ablation drop", colors["rand"], "--",
              lw * 0.75, error_mode)
    plot_line(ax, rows, "test_drop_random_ablation", "Test random-ablation drop", colors["rand"], "-",
              lw, error_mode, band=True)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy drop from original")
    ax.set_xscale(cfg["axis"]["x_scale"])
    ax.grid(True, alpha=cfg["grid"]["alpha"])
    suffix = f" ({error_mode} band)" if multi_seed else ""
    ax.set_title(f"x {symbol} y mod 97 - Robustness Drop{suffix}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(bottom=0.26)
    save_all(fig, os.path.join(out_dir, "perturb_drop_multiseed" if multi_seed else "perturb_drop"))
    plt.close(fig)


def plot_one(op_key, project_root, cfg, multi_seed, seeds, error_mode):
    op_dir, symbol = OPS[op_key]
    rows = load_rows(project_root, op_dir, multi_seed, seeds)
    out_dir = os.path.join(project_root, "results", op_dir, "perturb")
    os.makedirs(out_dir, exist_ok=True)
    if not rows:
        print(f"[SKIP] no perturb data found for {op_dir}")
        return
    plot_accuracy(symbol, rows, out_dir, cfg, multi_seed, error_mode)
    plot_drop(symbol, rows, out_dir, cfg, multi_seed, error_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--error", choices=["std", "sem"], default="std")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in ops:
        plot_one(op, project_root, cfg, args.multi_seed, args.seeds, args.error)


if __name__ == "__main__":
    main()
