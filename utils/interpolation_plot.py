#!/usr/bin/env python3
"""
Plot linear and curved interpolation metrics.

Single-seed mode reads:
    data/{op}/interpolation/interpolation_paths.csv
or legacy:
    data/{op}/interpolation.csv

Multi-seed mode reads:
    data/{op}/interpolation/interpolation_paths_multiseed.csv
or per-seed files:
    data/{op}/seed_{seed}/interpolation/interpolation_paths.csv
    data/seed_{seed}/{op}/interpolation/interpolation_paths.csv
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

PATH_STYLE = {
    "linear": {"color": "#2F6FBB", "label": "Linear"},
    "bezier": {"color": "#D55E00", "label": "Bezier curve"},
}


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


def seed_paths(project_root, op_dir, seeds, filename):
    data_root = os.path.join(project_root, "data")
    for seed in seeds:
        candidates = [
            os.path.join(data_root, op_dir, f"seed_{seed}", "interpolation", filename),
            os.path.join(data_root, f"seed_{seed}", op_dir, "interpolation", filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                yield path
                break


def load_paths(project_root, op_dir, multi_seed, seeds):
    data_dir = os.path.join(project_root, "data", op_dir)
    if multi_seed:
        aggregate = os.path.join(data_dir, "interpolation", "interpolation_paths_multiseed.csv")
        if os.path.exists(aggregate):
            return read_rows(aggregate)
        rows = []
        for path in seed_paths(project_root, op_dir, seeds, "interpolation_paths.csv"):
            rows.extend(read_rows(path))
        return rows

    preferred = os.path.join(data_dir, "interpolation", "interpolation_paths.csv")
    legacy = os.path.join(data_dir, "interpolation.csv")
    if os.path.exists(preferred):
        return read_rows(preferred)
    if os.path.exists(legacy):
        return read_rows(legacy)
    return []


def load_summary(project_root, op_dir, multi_seed, seeds):
    data_dir = os.path.join(project_root, "data", op_dir)
    if multi_seed:
        aggregate = os.path.join(data_dir, "interpolation", "interpolation_summary_multiseed.csv")
        if os.path.exists(aggregate):
            return read_rows(aggregate)
        rows = []
        for path in seed_paths(project_root, op_dir, seeds, "interpolation_summary.csv"):
            rows.extend(read_rows(path))
        return rows

    preferred = os.path.join(data_dir, "interpolation", "interpolation_summary.csv")
    return read_rows(preferred) if os.path.exists(preferred) else []


def save_all(fig, base):
    for fmt in ("pdf", "svg"):
        path = f"{base}.{fmt}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {path}")


def aggregate_curve(rows, path_name, col, error_mode):
    by_alpha = defaultdict(list)
    for row in rows:
        if row.get("path", "linear") != path_name:
            continue
        value = row.get(col, np.nan)
        if not np.isnan(value):
            by_alpha[row["alpha"]].append(value)
    xs, means, errs = [], [], []
    for alpha in sorted(by_alpha):
        vals = np.asarray(by_alpha[alpha], dtype=float)
        xs.append(alpha)
        means.append(float(vals.mean()))
        if len(vals) <= 1:
            errs.append(0.0)
        else:
            std = float(vals.std(ddof=1))
            errs.append(std / np.sqrt(len(vals)) if error_mode == "sem" else std)
    return np.asarray(xs), np.asarray(means), np.asarray(errs)


def available_paths(rows):
    return sorted(set(row.get("path", "linear") for row in rows))


def plot_paths(symbol, rows, out_dir, cfg, multi_seed, error_mode):
    paths = available_paths(rows)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    lw = cfg["line"]["linewidth"]
    panels = [
        ("test_acc", "Test Accuracy", (-0.05, 1.05)),
        ("test_loss", "Test Loss", None),
        ("test_margin", "Test Margin", None),
    ]
    for ax, (col, ylabel, ylim) in zip(axes, panels):
        for path_name in paths:
            style = PATH_STYLE.get(path_name, {"color": None, "label": path_name})
            alpha, mean, err = aggregate_curve(rows, path_name, col, error_mode)
            if len(alpha) == 0:
                continue
            ax.plot(alpha, mean, color=style["color"], linewidth=lw, label=style["label"])
            if multi_seed:
                ax.fill_between(alpha, mean - err, mean + err, color=style["color"], alpha=0.16, linewidth=0)
        if col == "test_margin":
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=cfg["grid"]["alpha"])
    axes[-1].set_xlabel(r"$\alpha$ (0 = memorization, 1 = generalization)")
    suffix = f" ({error_mode} band)" if multi_seed else ""
    axes[0].set_title(f"x {symbol} y mod 97 - Linear vs Curved Interpolation{suffix}")
    axes[0].legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    save_all(fig, os.path.join(out_dir, "interpolation_paths_multiseed" if multi_seed else "interpolation_paths"))
    plt.close(fig)


def summarize_from_rows(rows):
    summaries = []
    for path_name in available_paths(rows):
        path_rows = [r for r in rows if r.get("path", "linear") == path_name]
        start = min(path_rows, key=lambda r: r["alpha"])
        end = max(path_rows, key=lambda r: r["alpha"])
        min_endpoint_acc = min(start["test_acc"], end["test_acc"])
        max_endpoint_loss = max(start["test_loss"], end["test_loss"])
        min_endpoint_margin = min(start["test_margin"], end["test_margin"])
        summaries.append({
            "path": path_name,
            "test_acc_barrier": min_endpoint_acc - min(r["test_acc"] for r in path_rows),
            "test_loss_barrier": max(r["test_loss"] for r in path_rows) - max_endpoint_loss,
            "test_margin_barrier": min_endpoint_margin - min(r["test_margin"] for r in path_rows),
        })
    return summaries


def aggregate_summary(rows, metric, error_mode):
    by_path = defaultdict(list)
    for row in rows:
        value = row.get(metric, np.nan)
        if not np.isnan(value):
            by_path[row["path"]].append(value)
    stats = {}
    for path_name, vals in by_path.items():
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        if len(arr) <= 1:
            err = 0.0
        else:
            std = float(arr.std(ddof=1))
            err = std / np.sqrt(len(arr)) if error_mode == "sem" else std
        stats[path_name] = (mean, err)
    return stats


def plot_barriers(symbol, path_rows, summary_rows, out_dir, cfg, multi_seed, error_mode):
    summaries = summary_rows if summary_rows else summarize_from_rows(path_rows)
    if not summaries:
        return
    metrics = [
        ("test_acc_barrier", "Accuracy barrier"),
        ("test_loss_barrier", "Loss barrier"),
        ("test_margin_barrier", "Margin barrier"),
    ]
    paths = sorted(set(row["path"] for row in summaries))
    x = np.arange(len(metrics))
    width = 0.34 if len(paths) <= 2 else 0.8 / len(paths)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, path_name in enumerate(paths):
        vals, errs = [], []
        for metric, _ in metrics:
            if multi_seed:
                mean, err = aggregate_summary(summaries, metric, error_mode).get(path_name, (np.nan, 0.0))
            else:
                row = next(s for s in summaries if s["path"] == path_name)
                mean, err = row[metric], 0.0
            vals.append(mean)
            errs.append(err)
        style = PATH_STYLE.get(path_name, {"color": None, "label": path_name})
        offset = (idx - (len(paths) - 1) / 2) * width
        ax.bar(x + offset, vals, yerr=errs if multi_seed else None, width=width,
               color=style["color"], alpha=0.86, capsize=3, label=style["label"])
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Barrier height")
    suffix = f" ({error_mode} band)" if multi_seed else ""
    ax.set_title(f"x {symbol} y mod 97 - Interpolation Barrier Summary{suffix}")
    ax.grid(True, axis="y", alpha=cfg["grid"]["alpha"])
    ax.legend(frameon=False)
    fig.tight_layout()
    save_all(fig, os.path.join(out_dir, "interpolation_barriers_multiseed" if multi_seed else "interpolation_barriers"))
    plt.close(fig)


def plot_one(op_key, project_root, cfg, multi_seed, seeds, error_mode):
    op_dir, symbol = OPS[op_key]
    rows = load_paths(project_root, op_dir, multi_seed, seeds)
    summary_rows = load_summary(project_root, op_dir, multi_seed, seeds)
    out_dir = os.path.join(project_root, "results", op_dir, "interpolation")
    os.makedirs(out_dir, exist_ok=True)
    if not rows:
        print(f"[SKIP] no interpolation data found for {op_dir}")
        return
    plot_paths(symbol, rows, out_dir, cfg, multi_seed, error_mode)
    plot_barriers(symbol, rows, summary_rows, out_dir, cfg, multi_seed, error_mode)


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
