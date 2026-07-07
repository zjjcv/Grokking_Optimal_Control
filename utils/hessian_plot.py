#!/usr/bin/env python3
"""
Plot Hessian spectral probes alongside accuracy, and validate LLC against
Hessian-based degeneracy measures.

Reads:
    data/{op}/hessian.csv                data/{op}/metric.csv
    data/{op}/seed_{seed}/hessian.csv    data/{op}/seed_{seed}/metric.csv
    data/{op}/llc.csv or data/{op}/seed_{seed}/llc.csv (for correlation)
Writes:
    results/{op}/hessian_plot[_multiseed].{pdf,svg}
    results/{op}/hessian_llc_plot[_multiseed].{pdf,svg}

Usage:
    python utils/hessian_plot.py --operation add
    python utils/hessian_plot.py --operation all --multi-seed --seeds 0 1 2 --error std
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

DEFAULT_SEEDS = [0, 1, 2]


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


def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        data = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                try:
                    data[c].append(float(row[c]))
                except (ValueError, TypeError):
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


def plot_curve_with_band(ax, series, value_col, color, linestyle, linewidth, label, error_mode):
    steps, mean, error = aggregate_by_step(series, value_col, error_mode)
    if len(steps) == 0:
        return None
    line, = ax.plot(steps, mean, color=color, linestyle=linestyle,
                    linewidth=linewidth, label=label)
    ax.fill_between(steps, mean - error, mean + error, color=color, alpha=0.18, linewidth=0)
    return line


def plot_hessian_metrics(op_key, project_root, cfg, hess_series, metric_series,
                         multi_seed, error_mode):
    """Main panel: accuracy + lambda_max / trace(H); side panel: post-peak zoom."""
    op_dir, symbol = OPS[op_key]
    out_dir = os.path.join(project_root, "results", op_dir)
    os.makedirs(out_dir, exist_ok=True)

    lw = cfg["line"]["linewidth"]
    ls_train = cfg["line"]["train_linestyle"]
    ls_test = cfg["line"]["test_linestyle"]
    c_acc = cfg["color"]["accuracy"]
    leg = cfg["legend"]

    figsize = cfg["figure"]["figsize"]
    fig, (ax_acc, ax_zoom) = plt.subplots(
        1, 2, figsize=(figsize[0] * 1.35, figsize[1]),
        gridspec_kw={"width_ratios": [2.0, 1.0]})
    handles = []

    for col, label, linestyle in (("train_acc", "Train Acc", ls_train),
                                  ("test_acc", "Test Acc", ls_test)):
        line = plot_curve_with_band(ax_acc, metric_series, col, c_acc,
                                    linestyle, lw, label, error_mode)
        if line is not None:
            handles.append(line)

    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_h = ax_acc.twinx()
    for col, label, color in (("lambda_max", r"$\lambda_{\max}(H)$", "tab:purple"),
                              ("trace", r"$\mathrm{tr}(H)$", "tab:orange")):
        line = plot_curve_with_band(ax_h, hess_series, col, color, "-", lw, label, error_mode)
        if line is not None:
            handles.append(line)
    ax_h.set_ylabel("Hessian curvature")

    # side zoom panel: post-peak region where curvature is orders of magnitude smaller
    zoom_min_step = 3000
    for col, color in (("lambda_max", "tab:purple"), ("trace", "tab:orange")):
        steps, mean, error = aggregate_by_step(hess_series, col, error_mode)
        mask = steps >= zoom_min_step
        if not mask.any():
            continue
        ax_zoom.plot(steps[mask], mean[mask], color=color, linewidth=lw * 0.8)
        ax_zoom.fill_between(steps[mask], (mean - error)[mask], (mean + error)[mask],
                             color=color, alpha=0.18, linewidth=0)
    ax_zoom.set_xscale(cfg["axis"]["x_scale"])
    ax_zoom.grid(True, alpha=cfg["grid"]["alpha"])
    ax_zoom.set_xlabel("Step")
    ax_zoom.set_ylabel("Hessian curvature")
    ax_zoom.set_title(f"Zoom: step ≥ {zoom_min_step}")

    suffix = f" ({len(hess_series)} seeds, {error_mode})" if multi_seed else ""
    ax_acc.set_title(f"x {symbol} y mod 97{suffix}")

    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure, ncol=4,
               frameon=leg["frameon"], fontsize=leg["fontsize"])
    fig.subplots_adjust(bottom=0.2, wspace=0.35)

    base_name = "hessian_plot_multiseed" if multi_seed else "hessian_plot"
    for fmt in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"{base_name}.{fmt}")
        plt.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path}")
    plt.close()


def paired_llc_hessian(llc_series, hess_series):
    """Match LLC and Hessian rows on (seed, step); returns pooled arrays."""
    hess_by_key = {}
    for data in hess_series:
        seed = data.get("_seed", data["seed"][0] if "seed" in data else 0)
        for i, step in enumerate(data["step"]):
            hess_by_key[(seed, step)] = float(data["erank_pr"][i])

    llc_vals, erank_vals, steps = [], [], []
    for data in llc_series:
        seed = data.get("_seed", data["seed"][0] if "seed" in data else 0)
        for i, step in enumerate(data["step"]):
            key = (seed, step)
            if key in hess_by_key:
                llc_vals.append(float(data["llc_mean"][i]))
                erank_vals.append(hess_by_key[key])
                steps.append(step)
    return (np.asarray(llc_vals), np.asarray(erank_vals), np.asarray(steps))


def plot_llc_correlation(op_key, project_root, cfg, hess_series, llc_series,
                         multi_seed, error_mode):
    """(a) LLC vs erank_pr/2 trajectories; (b) scatter with Pearson/Spearman."""
    op_dir, symbol = OPS[op_key]
    out_dir = os.path.join(project_root, "results", op_dir)
    os.makedirs(out_dir, exist_ok=True)

    llc, erank, steps = paired_llc_hessian(llc_series, hess_series)
    if len(llc) < 3:
        print(f"[SKIP] not enough matched LLC/Hessian steps for {op_dir}")
        return

    lw = cfg["line"]["linewidth"]
    leg = cfg["legend"]
    c_llc = "tab:red"
    c_er = "tab:green"

    fig, (ax_traj, ax_scatter) = plt.subplots(
        1, 2, figsize=(cfg["figure"]["figsize"][0] * 1.4, cfg["figure"]["figsize"][1]))

    # (a) trajectories
    handles = []
    line = plot_curve_with_band(ax_traj, llc_series, "llc_mean", c_llc, "-", lw,
                                r"LLC $\hat\lambda$", error_mode)
    if line is not None:
        handles.append(line)
    ax_traj.set_xlabel("Step")
    ax_traj.set_ylabel(r"LLC $\hat\lambda$", color=c_llc)
    ax_traj.tick_params(axis="y", labelcolor=c_llc)
    ax_traj.set_xscale(cfg["axis"]["x_scale"])
    ax_traj.grid(True, alpha=cfg["grid"]["alpha"])

    ax_er = ax_traj.twinx()
    er_series = []
    for data in hess_series:
        er_series.append({
            "step": data["step"],
            "half_erank": [0.5 * float(v) for v in data["erank_pr"]],
        })
    line = plot_curve_with_band(ax_er, er_series, "half_erank", c_er, "-", lw,
                                r"$\mathrm{erank}(H)/2$", error_mode)
    if line is not None:
        handles.append(line)
    ax_er.set_ylabel(r"Hessian $\mathrm{erank}_{\mathrm{PR}}(H)/2$", color=c_er)
    ax_er.tick_params(axis="y", labelcolor=c_er)
    ax_traj.set_title("Trajectories")

    # (b) scatter, colored by log10(step)
    pearson_r, pearson_p = stats.pearsonr(llc, erank / 2.0)
    spearman_r, spearman_p = stats.spearmanr(llc, erank / 2.0)
    sc = ax_scatter.scatter(erank / 2.0, llc, c=np.log10(np.maximum(steps, 1)),
                            cmap="viridis", s=22, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax_scatter)
    cbar.set_label(r"$\log_{10}(\mathrm{step})$")
    ax_scatter.set_xlabel(r"$\mathrm{erank}_{\mathrm{PR}}(H)/2$")
    ax_scatter.set_ylabel(r"LLC $\hat\lambda$")
    ax_scatter.grid(True, alpha=cfg["grid"]["alpha"])
    ax_scatter.set_title(
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.1e})\n"
        f"Spearman ρ={spearman_r:.3f} (p={spearman_p:.1e})",
        fontsize=11,
    )

    suffix = f" ({len(hess_series)} seeds, {error_mode})" if multi_seed else ""
    fig.suptitle(f"x {symbol} y mod 97 — LLC vs Hessian degeneracy{suffix}",
                 fontweight=plt.rcParams.get("axes.titleweight", "bold"))

    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure, ncol=2,
               frameon=leg["frameon"], fontsize=leg["fontsize"])
    fig.subplots_adjust(bottom=0.22, top=0.82, wspace=0.35)

    base_name = "hessian_llc_plot_multiseed" if multi_seed else "hessian_llc_plot"
    for fmt in ("pdf", "svg"):
        out_path = os.path.join(out_dir, f"{base_name}.{fmt}")
        plt.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path}")
    plt.close()

    print(f"    matched points: {len(llc)} | Pearson={pearson_r:.4f} Spearman={spearman_r:.4f}")


def plot_one(op_key, project_root, cfg, multi_seed=False, seeds=None, error_mode="std"):
    op_dir, _ = OPS[op_key]
    data_dir = os.path.join(project_root, "data", op_dir)
    seeds = DEFAULT_SEEDS if seeds is None else seeds

    if multi_seed:
        hess_series = load_seed_series(project_root, op_dir, seeds, "hessian.csv")
        metric_series = load_seed_series(project_root, op_dir, seeds, "metric.csv")
        llc_series = load_seed_series(project_root, op_dir, seeds, "llc.csv")
    else:
        hess_csv = os.path.join(data_dir, "hessian.csv")
        metric_csv = os.path.join(data_dir, "metric.csv")
        llc_csv = os.path.join(data_dir, "llc.csv")
        hess_series = [read_csv(hess_csv)] if os.path.exists(hess_csv) else []
        metric_series = [read_csv(metric_csv)] if os.path.exists(metric_csv) else []
        llc_series = [read_csv(llc_csv)] if os.path.exists(llc_csv) else []
        # single-seed legacy CSVs may carry inconsistent seed columns; match on step only
        for data in hess_series + llc_series:
            data["_seed"] = 0

    if not hess_series:
        print(f"[SKIP] no hessian.csv found for {op_dir}")
        return
    if metric_series:
        plot_hessian_metrics(op_key, project_root, cfg, hess_series, metric_series,
                             multi_seed, error_mode)
    if llc_series:
        plot_llc_correlation(op_key, project_root, cfg, hess_series, llc_series,
                             multi_seed, error_mode)
    else:
        print(f"[SKIP] no llc.csv found for {op_dir}; correlation plot skipped")


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
        plot_one(op, project_root, cfg, args.multi_seed, args.seeds, args.error)


if __name__ == "__main__":
    main()
