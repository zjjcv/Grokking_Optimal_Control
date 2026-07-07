#!/usr/bin/env python3
"""
Plot spectral entropy alongside train/test accuracy.

Reads:
    data/{op}/spectral_entropy.csv
    data/{op}/metric.csv
    data/{op}/seed_{seed}/spectral_entropy.csv
    data/{op}/seed_{seed}/metric.csv
Writes:
    results/{op}/spectral_entropy_plot[_multiseed].{pdf,svg}

Usage:
    python utils/spectral_entropy_plot.py --operation add
    python utils/spectral_entropy_plot.py --operation all
"""

import os
import json
import csv
import argparse
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
            series.append(data)
    return series


def mean_parameter_entropy(data):
    param_cols = [
        c for c in data
        if c not in ("step", "seed", "run_dir") and not c.startswith("_")
    ]
    means = []
    for i in range(len(data["step"])):
        values = [float(data[col][i]) for col in param_cols]
        means.append(float(np.mean(values)))
    return {
        "step": data["step"],
        "spectral_entropy_mean": means,
    }


def aggregate_by_step(series, value_col, error_mode):
    by_step = defaultdict(list)
    for data in series:
        for step, value in zip(data["step"], data[value_col]):
            by_step[step].append(float(value))

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


def plot_curve_with_band(
    ax,
    series,
    value_col,
    color,
    linestyle,
    linewidth,
    label,
    error_mode,
):
    steps, mean, error = aggregate_by_step(series, value_col, error_mode)
    if len(steps) == 0:
        return None
    line, = ax.plot(
        steps,
        mean,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )
    ax.fill_between(
        steps,
        mean - error,
        mean + error,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    return line


def plot_one(
    op_key,
    project_root,
    cfg,
    multi_seed=False,
    seeds=None,
    error_mode="std",
):
    op_dir, symbol = OPS[op_key]
    data_dir = os.path.join(project_root, "data", op_dir)
    se_csv = os.path.join(data_dir, "spectral_entropy.csv")
    metric_csv = os.path.join(data_dir, "metric.csv")
    out_dir = os.path.join(project_root, "results", op_dir)
    os.makedirs(out_dir, exist_ok=True)

    seeds = DEFAULT_SEEDS if seeds is None else seeds
    if multi_seed:
        raw_se_series = load_seed_series(
            project_root,
            op_dir,
            seeds,
            "spectral_entropy.csv",
        )
        metric_data = load_seed_series(
            project_root,
            op_dir,
            seeds,
            "metric.csv",
        )
        if not raw_se_series:
            print(f"[SKIP] no multi-seed spectral entropy data found for {op_dir}")
            return
        se_data = [mean_parameter_entropy(data) for data in raw_se_series]
    else:
        if not os.path.exists(se_csv):
            print(f"[SKIP] {se_csv} not found")
            return
        if not os.path.exists(metric_csv):
            print(f"[SKIP] {metric_csv} not found")
            return
        raw_se_data = read_csv(se_csv)
        se_data = mean_parameter_entropy(raw_se_data)
        metric_data = read_csv(metric_csv)

    lw = cfg["line"]["linewidth"]
    ls_train = cfg["line"]["train_linestyle"]
    ls_test = cfg["line"]["test_linestyle"]
    c_acc = cfg["color"]["accuracy"]
    c_se = "tab:green"
    leg = cfg["legend"]

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = []

    # Left y-axis: Accuracy
    if multi_seed:
        for col, label, linestyle in (
            ("train_acc", "Train Acc", ls_train),
            ("test_acc", "Test Acc", ls_test),
        ):
            line = plot_curve_with_band(
                ax_acc,
                metric_data,
                col,
                c_acc,
                linestyle,
                lw,
                label,
                error_mode,
            )
            if line is not None:
                handles.append(line)
    else:
        steps = metric_data["step"]
        l1, = ax_acc.plot(steps, metric_data["train_acc"], color=c_acc,
                          linestyle=ls_train, linewidth=lw, label="Train Acc")
        l2, = ax_acc.plot(steps, metric_data["test_acc"], color=c_acc,
                          linestyle=ls_test, linewidth=lw, label="Test Acc")
        handles += [l1, l2]

    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    # Right y-axis: Spectral Entropy (mean across all components)
    ax_se = ax_acc.twinx()
    if multi_seed:
        line = plot_curve_with_band(
            ax_se,
            se_data,
            "spectral_entropy_mean",
            c_se,
            "-",
            lw,
            "Energy Spectral Entropy",
            error_mode,
        )
        if line is not None:
            handles.append(line)
    else:
        l3, = ax_se.plot(
            se_data["step"],
            se_data["spectral_entropy_mean"],
            color=c_se,
            linewidth=lw,
            label="Energy Spectral Entropy",
        )
        handles.append(l3)
    ax_se.set_ylabel("Energy Spectral Entropy", color=c_se)
    ax_se.tick_params(axis="y", labelcolor=c_se)

    suffix = f" ({len(se_data)} seeds, {error_mode})" if multi_seed else ""
    ax_acc.set_title(f"x {symbol} y mod 97{suffix}")

    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure, ncol=3,
               frameon=leg["frameon"], fontsize=leg["fontsize"])

    fig.subplots_adjust(bottom=0.18)
    base_name = (
        "spectral_entropy_plot_multiseed"
        if multi_seed
        else "spectral_entropy_plot"
    )
    for fmt in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"{base_name}.{fmt}")
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
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]

    for op in ops:
        plot_one(
            op,
            project_root,
            cfg,
            args.multi_seed,
            args.seeds,
            args.error,
        )


if __name__ == "__main__":
    main()
