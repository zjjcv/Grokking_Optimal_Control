#!/usr/bin/env python3
"""
Plot LLC robustness sweep with uncertainty bands.

Reads:
    data/{op}/llc_robustness/llc_robustness_raw.csv
    data/{op}/llc_robustness/llc_robustness_summary.csv
    data/{op}/metric.csv
Writes:
    results/{op}/llc_robustness/llc_robustness.svg

Usage:
    python utils/llc_robustness_plot.py --operation add
    python utils/llc_robustness_plot.py --operation all
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


def group_raw_curves(raw_data):
    grouped = defaultdict(lambda: {"step": [], "llc_mean": []})
    for step, lr, noise, seed, llc in zip(
        raw_data["step"],
        raw_data["lr"],
        raw_data["noise_level"],
        raw_data["seed"],
        raw_data["llc_mean"],
    ):
        key = (lr, noise, seed)
        grouped[key]["step"].append(step)
        grouped[key]["llc_mean"].append(llc)
    return grouped


def plot_accuracy(ax_acc, metric_data, cfg):
    handles = []
    if metric_data is None:
        return handles

    lw = cfg["line"]["linewidth"]
    c_acc = cfg["color"]["accuracy"]
    train, = ax_acc.plot(
        metric_data["step"],
        metric_data["train_acc"],
        color=c_acc,
        linestyle=cfg["line"]["train_linestyle"],
        linewidth=lw,
        label="Train Acc",
    )
    test, = ax_acc.plot(
        metric_data["step"],
        metric_data["test_acc"],
        color=c_acc,
        linestyle=cfg["line"]["test_linestyle"],
        linewidth=lw,
        label="Test Acc",
    )
    handles.extend([train, test])
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    return handles


def plot_one(op_key, project_root, cfg):
    op_dir, symbol = OPS[op_key]
    data_dir = os.path.join(project_root, "data", op_dir, "llc_robustness")
    raw_path = os.path.join(data_dir, "llc_robustness_raw.csv")
    summary_path = os.path.join(data_dir, "llc_robustness_summary.csv")
    metric_path = os.path.join(project_root, "data", op_dir, "metric.csv")
    out_dir = os.path.join(project_root, "results", op_dir, "llc_robustness")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"[SKIP] {raw_path} not found")
        return
    if not os.path.exists(summary_path):
        print(f"[SKIP] {summary_path} not found")
        return

    raw_data = read_csv(raw_path)
    summary = read_csv(summary_path)
    metric_data = read_csv(metric_path) if os.path.exists(metric_path) else None

    steps = np.array(summary["step"], dtype=float)
    mean = np.array(summary["llc_mean"], dtype=float)
    std = np.array(summary["llc_std"], dtype=float)

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = plot_accuracy(ax_acc, metric_data, cfg)

    ax_acc.set_xlabel("Step")
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_llc = ax_acc.twinx()
    for values in group_raw_curves(raw_data).values():
        order = np.argsort(values["step"])
        curve_steps = np.array(values["step"], dtype=float)[order]
        curve_llc = np.array(values["llc_mean"], dtype=float)[order]
        ax_llc.plot(curve_steps, curve_llc, color="tab:green", alpha=0.14, linewidth=0.8)

    ax_llc.fill_between(
        steps,
        mean - std,
        mean + std,
        color="tab:green",
        alpha=0.2,
        label="LLC +/- 1 std",
    )
    llc_line, = ax_llc.plot(
        steps,
        mean,
        color="tab:green",
        linewidth=cfg["line"]["linewidth"],
        marker="o",
        markersize=4,
        label="LLC mean",
    )
    handles.append(llc_line)

    ax_llc.set_ylabel("LLC", color="tab:green")
    ax_llc.tick_params(axis="y", labelcolor="tab:green")
    ax_acc.set_title(f"x {symbol} y mod 97 - LLC robustness")

    leg = cfg["legend"]
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc=leg["loc"],
        bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
        bbox_transform=fig.transFigure,
        ncol=min(len(handles), leg["ncol"]),
        frameon=leg["frameon"],
        fontsize=leg["fontsize"],
    )

    fig.subplots_adjust(bottom=0.2)
    out_path = os.path.join(out_dir, "llc_robustness.svg")
    plt.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[OK] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot LLC robustness sweep")
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    operations = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in operations:
        plot_one(op, project_root, cfg)


if __name__ == "__main__":
    main()
