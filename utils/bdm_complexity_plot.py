#!/usr/bin/env python3
"""
Plot BDM/CTM prediction-table complexity alongside train/test accuracy.

Reads:
    data/{op}/bdm_complexity.csv
    data/{op}/metric.csv
    data/{op}/seed_{seed}/bdm_complexity.csv
    data/{op}/seed_{seed}/metric.csv
Writes:
    results/{op}/bdm_complexity/bdm_complexity[_multiseed].{pdf,svg}
    results/{op}/bdm_complexity/bdm_entropy[_multiseed].{pdf,svg}

Usage:
    python utils/bdm_complexity_plot.py --operation add
    python utils/bdm_complexity_plot.py --operation all
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

DEFAULT_SEEDS = [0, 1, 2]

BDM_COLUMNS = [
    ("bdm_pred_per_cell", "BDM pred / cell", "tab:purple", "-"),
    ("bdm_correct_per_cell", "BDM correct / cell", "tab:green", "-"),
    ("bdm_pred_dlog_per_cell", "BDM pred dlog / cell", "tab:orange", "--"),
    ("bdm_correct_dlog_per_cell", "BDM correct dlog / cell", "tab:red", "--"),
]

ENTROPY_COLUMNS = [
    ("entropy_pred_per_cell", "Entropy pred / cell", "tab:purple", "-"),
    ("entropy_correct_per_cell", "Entropy correct / cell", "tab:green", "-"),
    ("entropy_pred_dlog_per_cell", "Entropy pred dlog / cell", "tab:orange", "--"),
    ("entropy_correct_dlog_per_cell", "Entropy correct dlog / cell", "tab:red", "--"),
]


def load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


def parse_float(value):
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
                data[c].append(parse_float(row[c]))
    return data


def has_values(values):
    return any(not math.isnan(v) for v in values)


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


def rows_from_multiseed_csv(path):
    if not os.path.exists(path):
        return []
    raw = read_csv(path)
    value_cols = [c for c in raw if c not in ("seed", "run_dir")]
    grouped = defaultdict(lambda: {c: [] for c in value_cols})
    for i, seed in enumerate(raw["seed"]):
        group = grouped[int(seed)]
        for col in value_cols:
            group[col].append(raw[col][i])
    return list(grouped.values())


def aggregate_by_step(series, value_col, error_mode):
    by_step = defaultdict(list)
    for data in series:
        if value_col not in data:
            continue
        for step, value in zip(data["step"], data[value_col]):
            if not math.isnan(value):
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
    steps, mean, err = aggregate_by_step(series, value_col, error_mode)
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
        mean - err,
        mean + err,
        color=color,
        alpha=0.16,
        linewidth=0,
    )
    return line


def add_accuracy_axis(ax_acc, metric_data, cfg, multi_seed=False, error_mode="std"):
    handles = []
    if not metric_data:
        return handles

    lw = cfg["line"]["linewidth"]
    c_acc = cfg["color"]["accuracy"]
    if multi_seed:
        for col, label, linestyle in (
            ("train_acc", "Train Acc", cfg["line"]["train_linestyle"]),
            ("test_acc", "Test Acc", cfg["line"]["test_linestyle"]),
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
        l1, = ax_acc.plot(
            metric_data["step"],
            metric_data["train_acc"],
            color=c_acc,
            linestyle=cfg["line"]["train_linestyle"],
            linewidth=lw,
            label="Train Acc",
        )
        l2, = ax_acc.plot(
            metric_data["step"],
            metric_data["test_acc"],
            color=c_acc,
            linestyle=cfg["line"]["test_linestyle"],
            linewidth=lw,
            label="Test Acc",
        )
        handles.extend([l1, l2])

    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    return handles


def filter_columns(columns, table):
    if table == "all":
        return columns
    return [c for c in columns if table in c[0]]


def plot_metric(
    op_key,
    symbol,
    bdm_data,
    metric_data,
    cfg,
    columns,
    ylabel,
    out_base,
    multi_seed=False,
    error_mode="std",
):
    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = add_accuracy_axis(
        ax_acc,
        metric_data,
        cfg,
        multi_seed,
        error_mode,
    )

    ax_acc.set_xlabel("Step")
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_right = ax_acc.twinx()
    for col, label, color, linestyle in columns:
        if multi_seed:
            line = plot_curve_with_band(
                ax_right,
                bdm_data,
                col,
                color,
                linestyle,
                cfg["line"]["linewidth"],
                label,
                error_mode,
            )
            if line is None:
                continue
        else:
            if col not in bdm_data or not has_values(bdm_data[col]):
                continue
            line, = ax_right.plot(
                bdm_data["step"],
                bdm_data[col],
                color=color,
                linestyle=linestyle,
                linewidth=cfg["line"]["linewidth"],
                label=label,
            )
        handles.append(line)

    ax_right.set_ylabel(ylabel, color="tab:purple")
    ax_right.tick_params(axis="y", labelcolor="tab:purple")
    suffix = f" ({error_mode} band)" if multi_seed else ""
    ax_acc.set_title(f"x {symbol} y mod 97 - BDM/CTM complexity{suffix}")

    leg = cfg["legend"]
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc=leg["loc"],
        bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
        bbox_transform=fig.transFigure,
        ncol=min(len(handles), 5),
        frameon=leg["frameon"],
        fontsize=leg["fontsize"],
    )

    fig.subplots_adjust(bottom=0.2)
    for fmt in ("pdf", "svg"):
        out_path = f"{out_base}.{fmt}"
        plt.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path}")
    plt.close()


def plot_one(
    op_key,
    project_root,
    cfg,
    section,
    table,
    multi_seed=False,
    seeds=None,
    error_mode="std",
):
    op_dir, symbol = OPS[op_key]
    data_dir = os.path.join(project_root, "data", op_dir)
    bdm_csv = os.path.join(data_dir, "bdm_complexity.csv")
    metric_csv = os.path.join(data_dir, "metric.csv")
    out_dir = os.path.join(project_root, "results", op_dir, "bdm_complexity")
    os.makedirs(out_dir, exist_ok=True)

    seeds = DEFAULT_SEEDS if seeds is None else seeds
    if multi_seed:
        bdm_data = load_seed_series(
            project_root,
            op_dir,
            seeds,
            "bdm_complexity.csv",
        )
        if not bdm_data:
            bdm_data = rows_from_multiseed_csv(
                os.path.join(data_dir, "bdm_complexity_multiseed.csv")
            )
        metric_data = load_seed_series(
            project_root,
            op_dir,
            seeds,
            "metric.csv",
        )
        if not bdm_data:
            print(f"[SKIP] no multi-seed BDM data found for {op_dir}")
            return
    else:
        if not os.path.exists(bdm_csv):
            print(f"[SKIP] {bdm_csv} not found")
            return
        bdm_data = read_csv(bdm_csv)
        metric_data = read_csv(metric_csv) if os.path.exists(metric_csv) else None

    suffix = "_multiseed" if multi_seed else ""

    if section in ("all", "bdm"):
        plot_metric(
            op_key,
            symbol,
            bdm_data,
            metric_data,
            cfg,
            filter_columns(BDM_COLUMNS, table),
            "BDM complexity per cell",
            os.path.join(out_dir, f"bdm_complexity{suffix}"),
            multi_seed,
            error_mode,
        )

    if section in ("all", "entropy"):
        plot_metric(
            op_key,
            symbol,
            bdm_data,
            metric_data,
            cfg,
            filter_columns(ENTROPY_COLUMNS, table),
            "Shannon entropy per cell",
            os.path.join(out_dir, f"bdm_entropy{suffix}"),
            multi_seed,
            error_mode,
        )


def main():
    parser = argparse.ArgumentParser(description="Plot BDM/CTM complexity")
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--section", choices=["all", "bdm", "entropy"], default="all")
    parser.add_argument("--table", choices=["all", "pred", "correct"], default="all",
                        help="Which table complexity to plot. Default: all")
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--error",
        choices=["std", "sem"],
        default="std",
        help="Error band across seeds.",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in ops:
        plot_one(
            op,
            project_root,
            cfg,
            args.section,
            args.table,
            args.multi_seed,
            args.seeds,
            args.error,
        )


if __name__ == "__main__":
    main()
