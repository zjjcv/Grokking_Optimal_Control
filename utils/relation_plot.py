#!/usr/bin/env python3
"""
Plot relation between head structure and CMA importance.

Reads:
    data/{op}/relation/relation_multiseed.csv
or per-seed files:
    data/{op}/seed_{seed}/relation/relation.csv
    data/seed_{seed}/{op}/relation/relation.csv

Writes:
    results/{op}/relation/relation_correlation.svg/pdf
    results/relation/relation_correlation_all.svg/pdf
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


def read_csv(path):
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


def seed_relation_paths(project_root, op_dir, seeds):
    data_root = os.path.join(project_root, "data")
    for seed in seeds:
        candidates = [
            os.path.join(data_root, op_dir, f"seed_{seed}", "relation", "relation.csv"),
            os.path.join(data_root, f"seed_{seed}", op_dir, "relation", "relation.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                yield path
                break


def load_relation_rows(project_root, op_dir, seeds):
    aggregate_path = os.path.join(project_root, "data", op_dir, "relation", "relation_multiseed.csv")
    if os.path.exists(aggregate_path):
        return read_csv(aggregate_path)

    rows = []
    for path in seed_relation_paths(project_root, op_dir, seeds):
        rows.extend(read_csv(path))
    return rows


def aggregate_by_step(rows, value_col, error_mode):
    by_step = defaultdict(list)
    for row in rows:
        value = row.get(value_col, np.nan)
        if np.isnan(value):
            continue
        by_step[row["step"]].append(value)

    steps, means, errors = [], [], []
    for step in sorted(by_step):
        values = np.asarray(by_step[step], dtype=float)
        steps.append(step)
        means.append(float(values.mean()))
        if len(values) <= 1:
            errors.append(0.0)
        else:
            std = float(values.std(ddof=1))
            errors.append(std / np.sqrt(len(values)) if error_mode == "sem" else std)
    return np.asarray(steps), np.asarray(means), np.asarray(errors)


def plot_one(op_key, project_root, cfg, seeds, error_mode, value_col):
    op_dir, symbol = OPS[op_key]
    rows = load_relation_rows(project_root, op_dir, seeds)
    if not rows:
        print(f"[SKIP] no relation data found for {op_dir}")
        return None

    steps, mean, err = aggregate_by_step(rows, value_col, error_mode)
    if len(steps) == 0:
        print(f"[SKIP] no finite {value_col} values for {op_dir}")
        return None

    out_dir = os.path.join(project_root, "results", op_dir, "relation")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=cfg["figure"]["figsize"])
    color = "tab:purple"
    ax.plot(steps, mean, color=color, linewidth=cfg["line"]["linewidth"], marker="o", label=value_col)
    ax.fill_between(steps, mean - err, mean + err, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xscale(cfg["axis"]["x_scale"])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"{value_col.title()} correlation")
    ax.set_title(f"x {symbol} y mod 97 - head structure/CMA relation ({error_mode})")
    ax.grid(True, alpha=cfg["grid"]["alpha"])
    ax.legend(frameon=False)

    base = os.path.join(out_dir, f"relation_{value_col}")
    for fmt in ("svg", "pdf"):
        path = f"{base}.{fmt}"
        plt.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {path}")
    plt.close()
    return steps, mean, err


def plot_all(ops, project_root, cfg, seeds, error_mode, value_col):
    out_dir = os.path.join(project_root, "results", "relation")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=cfg["figure"]["figsize"])
    colors = {
        "add": "#4C72B0",
        "sub": "#55A868",
        "mul": "#C44E52",
        "div": "#8172B2",
    }
    plotted = False
    for op_key in ops:
        op_dir, symbol = OPS[op_key]
        rows = load_relation_rows(project_root, op_dir, seeds)
        if not rows:
            continue
        steps, mean, err = aggregate_by_step(rows, value_col, error_mode)
        if len(steps) == 0:
            continue
        color = colors[op_key]
        ax.plot(steps, mean, color=color, linewidth=cfg["line"]["linewidth"], marker="o",
                label=f"x {symbol} y")
        ax.fill_between(steps, mean - err, mean + err, color=color, alpha=0.14, linewidth=0)
        plotted = True

    if not plotted:
        print("[SKIP] no relation data found for combined plot")
        plt.close()
        return

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xscale(cfg["axis"]["x_scale"])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"{value_col.title()} correlation")
    ax.set_title(f"Head OV-NFM structure vs CMA importance ({error_mode})")
    ax.grid(True, alpha=cfg["grid"]["alpha"])
    ax.legend(frameon=False, ncol=2)

    base = os.path.join(out_dir, f"relation_{value_col}_all")
    for fmt in ("svg", "pdf"):
        path = f"{base}.{fmt}"
        plt.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot head structure/CMA relation")
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--error", choices=["std", "sem"], default="std")
    parser.add_argument("--corr", choices=["pearson", "spearman"], default="pearson")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]

    for op in ops:
        plot_one(op, project_root, cfg, args.seeds, args.error, args.corr)
    if len(ops) > 1:
        plot_all(ops, project_root, cfg, args.seeds, args.error, args.corr)


if __name__ == "__main__":
    main()
