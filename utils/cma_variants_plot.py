#!/usr/bin/env python3
"""
Plot CMA ablation-variant robustness.

Left panel:  Spearman rank correlation between head-importance rankings
             (test accuracy drop) of each variant pair, across checkpoints.
Right panel: per-head test accuracy drop for all three variants at the last
             (grokked) checkpoint, as grouped bars.

Reads:
    data/{op}/cma/ablation_variants.csv
Writes:
    results/{op}/cma/cma_variants_plot.{pdf,svg}

Usage:
    python utils/cma_variants_plot.py --operation add
    python utils/cma_variants_plot.py --operation all
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OPS = {
    "add": ("x+y", "+"),
    "sub": ("x-y", "-"),
    "mul": ("x_mul_y", r"\times"),
    "div": ("x_div_y", r"\div"),
}

VARIANTS = ["mean", "zero", "resample"]
VARIANT_COLORS = {"mean": "tab:blue", "zero": "tab:orange", "resample": "tab:green"}
PAIRS = [("mean", "resample"), ("mean", "zero"), ("zero", "resample")]
PAIR_COLORS = ["tab:purple", "tab:brown", "tab:cyan"]


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


def plot_one(op_key, project_root, cfg):
    op_dir, symbol = OPS[op_key]
    csv_path = os.path.join(project_root, "data", op_dir, "cma", "ablation_variants.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found")
        return
    rows = read_rows(csv_path)

    out_dir = os.path.join(project_root, "results", op_dir, "cma")
    os.makedirs(out_dir, exist_ok=True)

    # drops[step][variant][head] = test_acc_drop
    drops = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        drops[int(r["step"])][r["variant"]][r["head"]] = float(r["test_acc_drop"])
    steps = sorted(drops)
    heads = sorted({r["head"] for r in rows})

    lw = cfg["line"]["linewidth"]
    leg = cfg["legend"]
    figsize = cfg["figure"]["figsize"]
    fig, (ax_corr, ax_bar) = plt.subplots(
        1, 2, figsize=(figsize[0] * 1.35, figsize[1]),
        gridspec_kw={"width_ratios": [1.0, 1.2]})

    # ---- left: Spearman rank correlation between variants over training ----
    for (va, vb), color in zip(PAIRS, PAIR_COLORS):
        xs, rhos = [], []
        for step in steps:
            a = [drops[step][va].get(h, np.nan) for h in heads]
            b = [drops[step][vb].get(h, np.nan) for h in heads]
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            rho, _ = stats.spearmanr(a, b)
            xs.append(step)
            rhos.append(rho)
        ax_corr.plot(xs, rhos, color=color, linewidth=lw, marker="o",
                     markersize=6, label=f"{va} vs {vb}")

    ax_corr.set_xscale(cfg["axis"]["x_scale"])
    ax_corr.set_ylim(-0.05, 1.05)
    ax_corr.set_xlabel("Step")
    ax_corr.set_ylabel(r"Spearman $\rho$ of head rankings")
    ax_corr.grid(True, alpha=cfg["grid"]["alpha"])
    ax_corr.set_title("Ranking consistency across variants")
    ax_corr.legend(frameon=leg["frameon"], fontsize=leg["fontsize"] - 1, loc="lower right")

    # ---- right: per-head drops at the final checkpoint ----
    last = steps[-1]
    x = np.arange(len(heads))
    width = 0.26
    for i, variant in enumerate(VARIANTS):
        vals = [drops[last][variant].get(h, np.nan) for h in heads]
        ax_bar.bar(x + (i - 1) * width, vals, width,
                   color=VARIANT_COLORS[variant], label=variant, edgecolor="black",
                   linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(heads, rotation=30)
    ax_bar.set_xlabel("Attention head")
    ax_bar.set_ylabel("Test accuracy drop")
    ax_bar.grid(True, axis="y", alpha=cfg["grid"]["alpha"])
    ax_bar.set_title(f"Per-head drop at step {last}")
    ax_bar.legend(frameon=leg["frameon"], fontsize=leg["fontsize"] - 1, loc="upper left")

    fig.suptitle(f"x {symbol} y mod 97 — ablation variant robustness")
    fig.subplots_adjust(wspace=0.28, top=0.86)

    for fmt in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"cma_variants_plot.{fmt}")
        plt.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path}")
    plt.close()

    # console summary
    rho_all = []
    for step in steps:
        for va, vb in PAIRS:
            a = [drops[step][va].get(h, np.nan) for h in heads]
            b = [drops[step][vb].get(h, np.nan) for h in heads]
            if not (np.isnan(a).any() or np.isnan(b).any()):
                rho_all.append(stats.spearmanr(a, b)[0])
    if rho_all:
        print(f"    mean Spearman over all steps/pairs: {np.mean(rho_all):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    apply_style(cfg)
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in ops:
        plot_one(op, project_root, cfg)


if __name__ == "__main__":
    main()
