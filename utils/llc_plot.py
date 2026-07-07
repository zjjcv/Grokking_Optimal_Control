#!/usr/bin/env python3
"""
Plot LLC and Train/Test Accuracy.

Single-seed mode reads:
  data/{op}/llc.csv
  data/{op}/metric.csv

Multi-seed mode auto-detects:
  data/{op}/seed_{seed}/llc.csv and metric.csv
  data/seed_{seed}/{op}/llc.csv and metric.csv
  data/{op}/llc_multiseed.csv as a fallback for LLC

The multi-seed plot draws mean curves with an error band across seeds.
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
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        data = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                try:
                    data[c].append(float(row[c]))
                except ValueError:
                    data[c].append(row[c])
    return data


def seed_run_dirs(project_root, op_dir, seeds):
    data_root = os.path.join(project_root, "data")
    for seed in seeds:
        candidates = [
            os.path.join(data_root, op_dir, f"seed_{seed}"),
            os.path.join(data_root, f"seed_{seed}", op_dir),
        ]
        for run_dir in candidates:
            if os.path.isdir(run_dir):
                yield seed, run_dir
                break


def load_seed_series(project_root, op_dir, seeds, filename):
    series = []
    for seed, run_dir in seed_run_dirs(project_root, op_dir, seeds):
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            data = read_csv(path)
            data["_seed"] = seed
            data["_path"] = path
            series.append(data)
    return series


def rows_from_multiseed_csv(path):
    if not os.path.exists(path):
        return []
    raw = read_csv(path)
    grouped = defaultdict(lambda: {"step": [], "llc_mean": [], "llc_std": []})
    for seed, step, mean, std in zip(raw["seed"], raw["step"], raw["llc_mean"], raw["llc_std"]):
        key = int(seed)
        grouped[key]["step"].append(step)
        grouped[key]["llc_mean"].append(mean)
        grouped[key]["llc_std"].append(std)
    rows = []
    for seed, data in sorted(grouped.items()):
        data["_seed"] = seed
        data["_path"] = path
        rows.append(data)
    return rows


def aggregate_by_step(series, value_col, error_mode):
    by_step = defaultdict(list)
    for data in series:
        for step, value in zip(data["step"], data[value_col]):
            by_step[step].append(value)

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


def plot_curve_with_band(ax, series, value_col, color, linestyle, linewidth, label, error_mode):
    steps, mean, err = aggregate_by_step(series, value_col, error_mode)
    if len(steps) == 0:
        return None
    line, = ax.plot(steps, mean, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    ax.fill_between(steps, mean - err, mean + err, color=color, alpha=0.18, linewidth=0)
    return line


def plot_one(op_key, project_root, cfg, multi_seed=False, seeds=None, error_mode="std"):
    op_dir, symbol = OPS[op_key]
    data_dir = os.path.join(project_root, "data", op_dir)
    llc_path = os.path.join(data_dir, "llc.csv")
    metric_path = os.path.join(data_dir, "metric.csv")
    seeds = DEFAULT_SEEDS if seeds is None else seeds

    if multi_seed:
        llc_series = load_seed_series(project_root, op_dir, seeds, "llc.csv")
        if not llc_series:
            llc_series = rows_from_multiseed_csv(os.path.join(data_dir, "llc_multiseed.csv"))
        metric_series = load_seed_series(project_root, op_dir, seeds, "metric.csv")
        if not metric_series and os.path.exists(metric_path):
            metric_series = [read_csv(metric_path)]
        if not llc_series:
            print(f"[SKIP] no multi-seed LLC data found for {op_dir}")
            return
    else:
        if not os.path.exists(llc_path):
            print(f"[SKIP] {llc_path} not found")
            return
        if not os.path.exists(metric_path):
            print(f"[SKIP] {metric_path} not found")
            return
        llc_data = read_csv(llc_path)
        metric_data = read_csv(metric_path)

    out_dir = os.path.join(project_root, "results", op_dir)
    os.makedirs(out_dir, exist_ok=True)

    lw = cfg["line"]["linewidth"]
    ls_train = cfg["line"]["train_linestyle"]
    ls_test = cfg["line"]["test_linestyle"]
    c_acc = cfg["color"]["accuracy"]
    c_llc = "tab:green"
    leg = cfg["legend"]

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])

    handles = []
    if multi_seed:
        h = plot_curve_with_band(ax_acc, metric_series, "train_acc", c_acc, ls_train, lw, "Train Acc", error_mode)
        if h is not None:
            handles.append(h)
        h = plot_curve_with_band(ax_acc, metric_series, "test_acc", c_acc, ls_test, lw, "Test Acc", error_mode)
        if h is not None:
            handles.append(h)
    else:
        l1, = ax_acc.plot(metric_data["step"], metric_data["train_acc"],
                          color=c_acc, linestyle=ls_train, linewidth=lw, label="Train Acc")
        l2, = ax_acc.plot(metric_data["step"], metric_data["test_acc"],
                          color=c_acc, linestyle=ls_test, linewidth=lw, label="Test Acc")
        handles.extend([l1, l2])

    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_llc = ax_acc.twinx()
    if multi_seed:
        h = plot_curve_with_band(ax_llc, llc_series, "llc_mean", c_llc, "-", lw, "LLC", error_mode)
        if h is not None:
            handles.append(h)
    else:
        l3, = ax_llc.plot(llc_data["step"], llc_data["llc_mean"],
                          color=c_llc, linewidth=lw, label="LLC")
        handles.append(l3)
    ax_llc.set_ylabel("LLC", color=c_llc)
    ax_llc.tick_params(axis="y", labelcolor=c_llc)

    suffix = f" ({len(seeds)} seeds, {error_mode})" if multi_seed else ""
    ax_acc.set_title(f"x {symbol} y mod 97{suffix}")

    fig.legend(handles, [h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure, ncol=3,
               frameon=leg["frameon"], fontsize=leg["fontsize"])

    fig.subplots_adjust(bottom=0.18)
    base_name = "llc_plot_multiseed" if multi_seed else "llc_plot"
    base = os.path.join(out_dir, base_name)
    for fmt in ("pdf", "svg"):
        path = f"{base}.{fmt}"
        plt.savefig(path, bbox_inches="tight")
        print(f"[OK] {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--error", choices=["std", "sem"], default="std",
                        help="Error band across seeds.")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]

    for op in ops:
        plot_one(op, project_root, cfg, args.multi_seed, args.seeds, args.error)


if __name__ == "__main__":
    main()
