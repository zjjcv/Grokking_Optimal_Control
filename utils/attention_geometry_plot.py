#!/usr/bin/env python3
"""
Plot attention-geometry group alignment across pathways.

Figure 1 (attention_geometry_alignment):
    Two panels, R_Fourier (higher = more group-aligned) and D_circ (lower =
    more circulant), one curve per pathway kind (OV-NFM, QK-NFM, V-NFM,
    positional QK score, empirical attention map), mean over heads with
    seed/head variability band.

Figure 2 (attention_map_evolution):
    Heatmap grid of empirical attention maps attmap_x (readout token ->
    x token) for all heads across representative checkpoints (seed 0 layout).

Reads:
    data/{op}[/seed_{s}]/attention_geometry/attention_geometry_metrics.csv
    data/{op}[/seed_{s}]/attention_geometry/attmap_x_l{l}_h{h}_step_{N}.npy
Writes:
    results/{op}/attention_geometry/attention_geometry_alignment.{pdf,svg}
    results/{op}/attention_geometry/attention_map_evolution.{pdf,svg}

Usage:
    python utils/attention_geometry_plot.py --operation add
    python utils/attention_geometry_plot.py --operation all --multi-seed --seeds 0 1 2
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

KINDS = [
    ("ovnfm", "OV-NFM", "tab:blue"),
    ("qknfm", "QK-NFM", "tab:orange"),
    ("vnfm", "V-NFM", "tab:green"),
    ("qkpos_x", "QK+pos score", "tab:purple"),
    ("attmap_x", "Empirical attention", "tab:red"),
]

HEATMAP_STEPS = [100, 1000, 5000, 30000, 90000]


def primitive_root(p):
    phi = p - 1
    factors = set()
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"No primitive root found for p={p}")


def maybe_dlog_reorder(mat, op_key):
    """Reorder rows/cols by powers of the primitive root for mul/div,
    consistent with the metric computation in src/attention_geometry.py."""
    if op_key not in ("mul", "div"):
        return mat
    p = mat.shape[0]
    order = []
    val = 1
    g = primitive_root(p)
    for _ in range(p - 1):
        order.append(val)
        val = (val * g) % p
    return mat[np.ix_(order, order)]


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


def seed_run_dirs(project_root, op_dir, seeds, multi_seed):
    data_root = os.path.join(project_root, "data")
    if not multi_seed:
        yield None, os.path.join(data_root, op_dir)
        return
    for seed in seeds:
        candidates = [
            os.path.join(data_root, op_dir, f"seed_{seed}"),
            os.path.join(data_root, f"seed_{seed}", op_dir),
        ]
        for run_dir in candidates:
            if os.path.isdir(run_dir):
                yield seed, run_dir
                break


def read_metrics(run_dir):
    path = os.path.join(run_dir, "attention_geometry", "attention_geometry_metrics.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def plot_alignment(op_key, project_root, cfg, multi_seed, seeds, error_mode):
    op_dir, symbol = OPS[op_key]
    out_dir = os.path.join(project_root, "results", op_dir, "attention_geometry")
    os.makedirs(out_dir, exist_ok=True)

    # values[kind][metric][step] = list over (seed, head) of metric value
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    found = False
    for seed, run_dir in seed_run_dirs(project_root, op_dir, seeds, multi_seed):
        for r in read_metrics(run_dir):
            found = True
            step = int(r["step"])
            for metric in ("R_Fourier", "D_circ"):
                values[r["kind"]][metric][step].append(float(r[metric]))
    if not found:
        print(f"[SKIP] no attention_geometry metrics for {op_dir}")
        return

    lw = cfg["line"]["linewidth"]
    leg = cfg["legend"]
    figsize = cfg["figure"]["figsize"]
    fig, (ax_rf, ax_dc) = plt.subplots(1, 2, figsize=(figsize[0] * 1.4, figsize[1]))

    handles = []
    for ax, metric, ylabel in ((ax_rf, "R_Fourier", r"$R_{\mathrm{Fourier}}$ (higher = aligned)"),
                               (ax_dc, "D_circ", r"$D_{\mathrm{circ}}$ (lower = circulant)")):
        for kind, label, color in KINDS:
            if kind not in values:
                continue
            steps = sorted(values[kind][metric])
            means = np.array([np.mean(values[kind][metric][s]) for s in steps])
            stds = np.array([np.std(values[kind][metric][s]) for s in steps])
            errs = stds / np.sqrt([len(values[kind][metric][s]) for s in steps]) \
                if error_mode == "sem" else stds
            line, = ax.plot(steps, means, color=color, linewidth=lw, marker="o",
                            markersize=5, label=label)
            ax.fill_between(steps, means - errs, means + errs, color=color,
                            alpha=0.15, linewidth=0)
            if ax is ax_rf:
                handles.append(line)
        ax.set_xscale(cfg["axis"]["x_scale"])
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=cfg["grid"]["alpha"])

    ax_rf.set_title("Fourier-line energy")
    ax_dc.set_title("Circulant deviation")

    n_seeds = len(set(seeds)) if multi_seed else 1
    fig.suptitle(f"x ${symbol}$ y mod 97 — group alignment across attention pathways"
                 f" ({n_seeds} seed{'s' if n_seeds > 1 else ''}, mean over heads)")
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure, ncol=len(handles),
               frameon=leg["frameon"], fontsize=leg["fontsize"] - 1)
    fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.28)

    for fmt in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"attention_geometry_alignment.{fmt}")
        plt.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path}")
    plt.close()


def plot_attention_maps(op_key, project_root, cfg, multi_seed, seeds):
    op_dir, symbol = OPS[op_key]
    out_dir = os.path.join(project_root, "results", op_dir, "attention_geometry")
    os.makedirs(out_dir, exist_ok=True)

    run_dir = next(iter(seed_run_dirs(project_root, op_dir, seeds, multi_seed)))[1]
    geo_dir = os.path.join(run_dir, "attention_geometry")
    if not os.path.isdir(geo_dir):
        print(f"[SKIP] {geo_dir} not found")
        return

    heads = [(l, h) for l in range(2) for h in range(4)]
    steps = [s for s in HEATMAP_STEPS
             if os.path.exists(os.path.join(geo_dir, f"attmap_x_l0_h0_step_{s}.npy"))]
    if not steps:
        print(f"[SKIP] no attmap npy files in {geo_dir}")
        return

    n_rows, n_cols = len(heads), len(steps)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.1 * n_cols + 1.2, 2.0 * n_rows),
                             squeeze=False)

    for i, (l, h) in enumerate(heads):
        for j, step in enumerate(steps):
            ax = axes[i][j]
            path = os.path.join(geo_dir, f"attmap_x_l{l}_h{h}_step_{step}.npy")
            if not os.path.exists(path):
                ax.axis("off")
                continue
            mat = maybe_dlog_reorder(np.load(path), op_key)
            im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"step {step}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"l{l}_h{h}", fontsize=11)

    cbar = fig.colorbar(im, ax=[axes[i][j] for i in range(n_rows) for j in range(n_cols)],
                        fraction=0.02, pad=0.02)
    cbar.set_label("Attention weight (readout → x)")
    cbar.outline.set_linewidth(plt.rcParams["axes.linewidth"] * 0.8)

    reorder_note = ", dlog-reordered" if op_key in ("mul", "div") else ""
    fig.suptitle(f"x ${symbol}$ y mod 97 — empirical attention maps "
                 f"(rows: heads, axes: y horizontal / x vertical{reorder_note})")

    for fmt in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"attention_map_evolution.{fmt}")
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
    apply_style(cfg)
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in ops:
        plot_alignment(op, project_root, cfg, args.multi_seed, args.seeds, args.error)
        plot_attention_maps(op, project_root, cfg, args.multi_seed, args.seeds)


if __name__ == "__main__":
    main()
