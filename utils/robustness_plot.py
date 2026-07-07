#!/usr/bin/env python3
"""
Plot hyperparameter robustness sweep results.

Reads:
    data/x+y/robustness/{cfg}/seed_{s}/metric.csv
    data/x+y/robustness/signatures_summary.csv
    data/x+y/robustness/relation_stats_pooled.csv

Writes:
    results/x+y/robustness/robustness_curves.svg
    results/x+y/robustness/grokking_steps.svg
    results/x+y/robustness/signatures_summary.svg
    results/x+y/robustness/relation_pooled.svg

Usage:
    python utils/robustness_plot.py
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from robustness_common import (
    DEFAULT_SEEDS,
    discover_robustness_runs,
    load_metric_csv,
    robustness_root,
)


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
        "font.weight": weight,
        "axes.labelweight": weight,
        "axes.titleweight": weight,
        "figure.titleweight": weight,
        "font.size": style.get("font_size", 12),
    })


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in ("svg", "pdf"):
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {path}")


def plot_curves(project_root, cfg, seeds):
    runs = discover_robustness_runs(project_root, "add", seeds=seeds)
    if not runs:
        print("[SKIP] no robustness runs for curves")
        return

    by_config = defaultdict(list)
    for config_name, seed, run_dir in runs:
        rows = load_metric_csv(os.path.join(run_dir, "metric.csv"))
        if rows:
            by_config[config_name].append(rows)

    if not by_config:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(by_config)))
    lw = cfg["line"]["linewidth"]

    for (config_name, seed_rows), color in zip(sorted(by_config.items()), colors):
        # mean over seeds
        by_step = defaultdict(list)
        for rows in seed_rows:
            for row in rows:
                by_step[row["step"]].append(row["test_acc"])
        steps = sorted(by_step)
        mean = [np.mean(by_step[s]) for s in steps]
        std = [np.std(by_step[s]) for s in steps]
        ax.plot(steps, mean, color=color, linewidth=lw, label=config_name)
        if len(seed_rows) > 1:
            ax.fill_between(steps, np.array(mean) - std, np.array(mean) + std,
                            color=color, alpha=0.12, linewidth=0)

    ax.set_xscale(cfg["axis"]["x_scale"])
    ax.set_ylim(*cfg["axis"]["acc_ylim"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Test accuracy")
    ax.set_title("x + y mod 97 — test accuracy across hyperparameter configs")
    ax.grid(True, alpha=cfg["grid"]["alpha"])
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    save_fig(fig, os.path.join(project_root, "results", "x+y", "robustness"), "robustness_curves")
    plt.close()


def plot_grokking_steps(project_root, cfg):
    summary_path = os.path.join(robustness_root(project_root, "add"), "signatures_summary.csv")
    rows = load_csv(summary_path)
    if not rows:
        print("[SKIP] no signatures_summary.csv")
        return

    by_config = defaultdict(list)
    for row in rows:
        g = row.get("grokking_step", "")
        if g == "":
            continue
        by_config[row["config"]].append(int(float(g)))

    if not by_config:
        print("[SKIP] no grokking steps detected")
        return

    configs = sorted(by_config)
    means = [np.mean(by_config[c]) for c in configs]
    stds = [np.std(by_config[c]) if len(by_config[c]) > 1 else 0 for c in configs]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(configs))
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.85,
           edgecolor="black", linewidth=cfg["style"].get("spine_linewidth", 1.8) * 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.set_ylabel("Grokking step (test acc ≥ 0.9)")
    ax.set_title("x + y mod 97 — grokking transition by config")
    ax.grid(True, axis="y", alpha=cfg["grid"]["alpha"])
    save_fig(fig, os.path.join(project_root, "results", "x+y", "robustness"), "grokking_steps")
    plt.close()


def plot_signatures(project_root, cfg):
    summary_path = os.path.join(robustness_root(project_root, "add"), "signatures_summary.csv")
    rows = load_csv(summary_path)
    if not rows:
        return

    configs = sorted(set(r["config"] for r in rows))
    final_acc = []
    peak_erank = []
    late_pearson = []
    for c in configs:
        subset = [r for r in rows if r["config"] == c]
        final_acc.append(np.mean([float(r["final_test_acc"]) for r in subset if r["final_test_acc"]]))
        eranks = [float(r["peak_erank_pr"]) for r in subset if r.get("peak_erank_pr", "")]
        peak_erank.append(np.mean(eranks) if eranks else np.nan)
        pears = [float(r["late_pearson"]) for r in subset if r.get("late_pearson", "")]
        late_pearson.append(np.mean(pears) if pears else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(configs))
    lw = cfg["style"].get("spine_linewidth", 1.8)

    for ax, values, title, ylabel in [
        (axes[0], final_acc, "Final test accuracy", "Accuracy"),
        (axes[1], peak_erank, "Peak Hessian erank_pr", "erank_pr"),
        (axes[2], late_pearson, "Late-step Pearson (structure vs CMA)", "Pearson r"),
    ]:
        ax.bar(x, values, color="tab:purple", alpha=0.85, edgecolor="black", linewidth=lw * 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=cfg["grid"]["alpha"])

    fig.suptitle("x + y mod 97 — signature summary across configs")
    fig.tight_layout()
    save_fig(fig, os.path.join(project_root, "results", "x+y", "robustness"), "signatures_summary")
    plt.close()


def plot_relation_pooled(project_root, cfg):
    path = os.path.join(robustness_root(project_root, "add"), "relation_stats_pooled.csv")
    rows = load_csv(path)
    if not rows:
        print("[SKIP] no relation_stats_pooled.csv")
        return

    pools = sorted(set(r["pool"] for r in rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"baseline_only": "#4C72B0", "robustness_only": "#C44E52", "all_pooled": "#55A868"}
    lw = cfg["line"]["linewidth"]

    for pool in pools:
        subset = sorted([r for r in rows if r["pool"] == pool], key=lambda r: int(r["step"]))
        steps = [int(r["step"]) for r in subset]
        mean = [float(r["pearson_boot_mean"]) for r in subset]
        lo = [float(r["pearson_ci_lo"]) for r in subset]
        hi = [float(r["pearson_ci_hi"]) for r in subset]
        ax.plot(steps, mean, marker="o", linewidth=lw, label=pool, color=colors.get(pool, "gray"))
        ax.fill_between(steps, lo, hi, alpha=0.15, color=colors.get(pool, "gray"), linewidth=0)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xscale(cfg["axis"]["x_scale"])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("Pooled Pearson r (bootstrap 95% CI)")
    ax.set_title("x + y mod 97 — pooled head structure/CMA correlation")
    ax.legend(frameon=False)
    ax.grid(True, alpha=cfg["grid"]["alpha"])
    save_fig(fig, os.path.join(project_root, "results", "x+y", "robustness"), "relation_pooled")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot robustness sweep results")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    apply_style(cfg)

    plot_curves(project_root, cfg, args.seeds)
    plot_grokking_steps(project_root, cfg)
    plot_signatures(project_root, cfg)
    plot_relation_pooled(project_root, cfg)


if __name__ == "__main__":
    main()
