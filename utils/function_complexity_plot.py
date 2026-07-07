#!/usr/bin/env python3
"""
Plot cross-validated learned-function complexity proxies.

Reads:
    data/{op}/function_complexity/function_complexity_bdm.csv
    data/{op}/function_complexity/function_complexity_compression_fourier.csv
    data/{op}/metric.csv
Writes:
    results/{op}/function_complexity/function_complexity_bdm.svg
    results/{op}/function_complexity/function_complexity_compression.svg
    results/{op}/function_complexity/function_complexity_fourier_topk.svg
    results/{op}/function_complexity/function_complexity_fourier_effective.svg

Usage:
    python utils/function_complexity_plot.py --operation all
    python utils/function_complexity_plot.py --operation mul --target pred
"""

import argparse
import csv
import json
import math
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

COMPLEXITY_COLORS = [
    "tab:purple",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
]


def load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


def parse_value(value):
    if value == "" or value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        data = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                data[c].append(parse_value(row[c]))
    return data


def has_value(x):
    return isinstance(x, float) and not math.isnan(x)


def filter_rows(data, target, encoding, coord, bdm_shape=None):
    indices = []
    n = len(data["step"])
    for i in range(n):
        if target != "all" and data["target"][i] != target:
            continue
        if encoding != "all" and data["encoding"][i] != encoding:
            continue
        if coord != "all" and data["coord"][i] != coord:
            continue
        if bdm_shape is not None and data.get("bdm_shape", [None] * n)[i] != bdm_shape:
            continue
        indices.append(i)
    return indices


def series_by_label(data, value_col, indices, include_shape=False):
    grouped = defaultdict(lambda: {"step": [], value_col: []})
    for i in indices:
        value = data[value_col][i]
        if not has_value(value):
            continue
        parts = [str(data["coord"][i]), str(data["target"][i]), str(data["encoding"][i])]
        if include_shape and "bdm_shape" in data:
            parts.append(str(data["bdm_shape"][i]))
        label = " / ".join(parts)
        grouped[label]["step"].append(data["step"][i])
        grouped[label][value_col].append(value)
    return dict(sorted(grouped.items()))


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


def plot_metric(data, metric_data, cfg, symbol, value_col, ylabel, title, out_path,
                target, encoding, coord, bdm_shape=None, include_shape=False):
    indices = filter_rows(data, target, encoding, coord, bdm_shape=bdm_shape)
    grouped = series_by_label(data, value_col, indices, include_shape=include_shape)
    if not grouped:
        print(f"[SKIP] no rows for {out_path}")
        return

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = plot_accuracy(ax_acc, metric_data, cfg)
    ax_acc.set_xlabel("Step")
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_right = ax_acc.twinx()
    for idx, (label, values) in enumerate(grouped.items()):
        order = np.argsort(values["step"])
        steps = np.array(values["step"])[order]
        ys = np.array(values[value_col])[order]
        line, = ax_right.plot(
            steps,
            ys,
            color=COMPLEXITY_COLORS[idx % len(COMPLEXITY_COLORS)],
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=label,
        )
        handles.append(line)

    ax_right.set_ylabel(ylabel)
    ax_acc.set_title(f"x {symbol} y mod 97 - {title}")

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
    fig.subplots_adjust(bottom=0.24)
    plt.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[OK] {out_path}")


def topk_column(cf_data):
    for col in cf_data:
        if col.startswith("fourier_top") and col.endswith("_energy") and col != "fourier_top1_energy":
            return col
    return "fourier_top1_energy"


def process_operation(op_key, project_root, cfg, args):
    op_dir, symbol = OPS[op_key]
    data_dir = os.path.join(project_root, "data", op_dir, "function_complexity")
    bdm_path = os.path.join(data_dir, "function_complexity_bdm.csv")
    cf_path = os.path.join(data_dir, "function_complexity_compression_fourier.csv")
    metric_path = os.path.join(project_root, "data", op_dir, "metric.csv")
    out_dir = os.path.join(project_root, "results", op_dir, "function_complexity")
    os.makedirs(out_dir, exist_ok=True)

    metric_data = read_csv(metric_path) if os.path.exists(metric_path) else None

    if os.path.exists(bdm_path):
        bdm = read_csv(bdm_path)
        plot_metric(
            bdm,
            metric_data,
            cfg,
            symbol,
            "bdm_per_cell",
            "BDM / cell",
            "BDM/CTM block sensitivity",
            os.path.join(out_dir, "function_complexity_bdm.svg"),
            args.target,
            args.encoding,
            args.coord,
            bdm_shape=args.bdm_shape if args.bdm_shape != "all" else None,
            include_shape=True,
        )
    else:
        print(f"[SKIP] {bdm_path} not found")

    if os.path.exists(cf_path):
        cf = read_csv(cf_path)
        plot_metric(
            cf,
            metric_data,
            cfg,
            symbol,
            "zlib_bits_per_cell",
            "zlib compressed bits / cell",
            "compression complexity",
            os.path.join(out_dir, "function_complexity_compression.svg"),
            args.target,
            args.encoding,
            args.coord,
        )

        topk_col = topk_column(cf)
        plot_metric(
            cf,
            metric_data,
            cfg,
            symbol,
            topk_col,
            "Fourier top-k energy",
            "Fourier sparsity",
            os.path.join(out_dir, "function_complexity_fourier_topk.svg"),
            args.target,
            args.encoding,
            args.coord,
        )

        plot_metric(
            cf,
            metric_data,
            cfg,
            symbol,
            "fourier_effective_fraction",
            "Fourier effective support fraction",
            "Fourier effective support",
            os.path.join(out_dir, "function_complexity_fourier_effective.svg"),
            args.target,
            args.encoding,
            args.coord,
        )
    else:
        print(f"[SKIP] {cf_path} not found")


def main():
    parser = argparse.ArgumentParser(description="Plot learned-function complexity proxies")
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--target", choices=["all", "pred", "correct"], default="pred")
    parser.add_argument("--encoding", choices=["all", "bitplane", "onehot", "binary"], default="all")
    parser.add_argument("--coord", choices=["all", "natural", "dlog"], default="all")
    parser.add_argument("--bdm-shape", choices=["all", "2x2", "3x3", "4x4"], default="all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    operations = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op_key in operations:
        process_operation(op_key, project_root, cfg, args)


if __name__ == "__main__":
    main()
