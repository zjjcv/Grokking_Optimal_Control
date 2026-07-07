#!/usr/bin/env python3
"""
Compute and plot circulant deviation and Fourier-line energy for AGOP/NFM matrices.

For each square matrix M, this computes

    D_circ(M) = min(
        ||M - Pi_diag(M)||_F / (||M||_F + eps),
        ||M - Pi_anti(M)||_F / (||M||_F + eps)
    )

and

    R_Fourier(M) = max(
        sum_{u+v=0} |Mhat_uv|^2 / (sum_{u,v} |Mhat_uv|^2 + eps),
        sum_{u-v=0} |Mhat_uv|^2 / (sum_{u,v} |Mhat_uv|^2 + eps)
    ).

For mul/div, matrices are first restricted to nonzero tokens and reordered by
discrete-log order over Z_p^*, so the analyzed matrices are 96x96 for p=97.

Reads:
    data/{op}/agop/agop_step_{step}.npy
    data/{op}/nfm/nfm_{component}_step_{step}.npy
Writes:
    data/{op}/circulant_fourier/agop_circulant_fourier.csv
    data/{op}/circulant_fourier/nfm_circulant_fourier.csv
    results/{op}/circulant_fourier/*.svg

Usage:
    python utils/circulant_fourier_plot.py --operation all
    python utils/circulant_fourier_plot.py --operation mul --source nfm
"""

import argparse
import csv
import json
import os
import re
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

AGOP_BLOCKS = {
    "Gxx": lambda agop, p: agop[:p, :p],
    "Gxy": lambda agop, p: agop[:p, p:],
    "Gyx": lambda agop, p: agop[p:, :p],
    "Gyy": lambda agop, p: agop[p:, p:],
}

NFM_GROUPS = {
    "embedding": ["embedding"],
    "unembedding": ["unembedding"],
    "mlp_in": ["mlp_in_l0", "mlp_in_l1"],
    "mlp_out": ["mlp_out_l0", "mlp_out_l1"],
    "qk": [f"qk_l{l}_h{h}" for l in range(2) for h in range(4)],
    "ov": [f"ov_l{l}_h{h}" for l in range(2) for h in range(4)],
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


def load_seed_csvs(project_root, op_dir, seeds, relative_path):
    series = []
    for seed, run_dir in seed_run_dirs(project_root, op_dir, seeds):
        path = os.path.join(run_dir, relative_path)
        if os.path.exists(path):
            data = read_csv(path)
            data["_seed"] = seed
            data["_path"] = path
            series.append(data)
    return series


def primitive_root(p):
    if p == 2:
        return 1
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


def dlog_nonzero_order(p):
    g = primitive_root(p)
    order = []
    val = 1
    for _ in range(p - 1):
        order.append(val)
        val = (val * g) % p
    return np.array(order, dtype=int)


def maybe_dlog_reorder(mat, op_key, p):
    if op_key not in ("mul", "div"):
        return mat
    order = dlog_nonzero_order(p)
    return mat[np.ix_(order, order)]


def diag_projection(mat):
    n = mat.shape[0]
    proj = np.zeros_like(mat, dtype=np.float64)
    rows = np.arange(n)
    for offset in range(n):
        cols = (rows + offset) % n
        proj[rows, cols] = mat[rows, cols].mean()
    return proj


def anti_projection(mat):
    n = mat.shape[0]
    proj = np.zeros_like(mat, dtype=np.float64)
    rows = np.arange(n)
    for total in range(n):
        cols = (total - rows) % n
        proj[rows, cols] = mat[rows, cols].mean()
    return proj


def circulant_deviation(mat, eps):
    m = np.asarray(mat, dtype=np.float64)
    norm = np.linalg.norm(m, ord="fro")
    d_diag = np.linalg.norm(m - diag_projection(m), ord="fro") / (norm + eps)
    d_anti = np.linalg.norm(m - anti_projection(m), ord="fro") / (norm + eps)
    return min(d_diag, d_anti), d_diag, d_anti


def fourier_energy(mat, eps):
    m = np.asarray(mat, dtype=np.float64)
    spectrum = np.fft.fft2(m)
    energy = np.abs(spectrum) ** 2
    total = float(energy.sum())
    n = m.shape[0]
    idx = np.arange(n)

    line_sum_zero = float(energy[idx, (-idx) % n].sum()) / (total + eps)
    line_diff_zero = float(energy[idx, idx].sum()) / (total + eps)
    return max(line_sum_zero, line_diff_zero), line_sum_zero, line_diff_zero


def matrix_metrics(mat, eps):
    d_circ, d_diag, d_anti = circulant_deviation(mat, eps)
    r_fourier, r_sum_zero, r_diff_zero = fourier_energy(mat, eps)
    return {
        "matrix_size": mat.shape[0],
        "D_circ": d_circ,
        "D_diag": d_diag,
        "D_anti": d_anti,
        "R_Fourier": r_fourier,
        "R_sum_zero": r_sum_zero,
        "R_diff_zero": r_diff_zero,
    }


def nfm_group(component):
    for group, components in NFM_GROUPS.items():
        if component in components:
            return group
    return "other"


def compute_agop(op_key, project_root, p, eps, run_dir=None):
    op_dir, _ = OPS[op_key]
    base_dir = run_dir or os.path.join(project_root, "data", op_dir)
    agop_dir = os.path.join(base_dir, "agop")
    out_dir = os.path.join(base_dir, "circulant_fourier")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "agop_circulant_fourier.csv")

    if not os.path.isdir(agop_dir):
        print(f"[SKIP] {agop_dir} not found")
        return None

    files = sorted(
        [f for f in os.listdir(agop_dir) if f.startswith("agop_step_") and f.endswith(".npy")],
        key=lambda f: int(f.split("_step_")[1].split(".")[0]),
    )
    if not files:
        print(f"[SKIP] no AGOP files in {agop_dir}")
        return None

    fields = [
        "step",
        "source",
        "component",
        "group",
        "matrix_size",
        "D_circ",
        "D_diag",
        "D_anti",
        "R_Fourier",
        "R_sum_zero",
        "R_diff_zero",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for filename in files:
            step = int(filename.split("_step_")[1].split(".")[0])
            agop = np.load(os.path.join(agop_dir, filename))
            for block_name, getter in AGOP_BLOCKS.items():
                block = maybe_dlog_reorder(getter(agop, p), op_key, p)
                metrics = matrix_metrics(block, eps)
                row = {
                    "step": step,
                    "source": "agop",
                    "component": block_name,
                    "group": "agop",
                    **metrics,
                }
                writer.writerow(format_row(row))

    print(f"[OK] {out_csv}")
    return out_csv


def compute_nfm(op_key, project_root, p, eps, run_dir=None):
    op_dir, _ = OPS[op_key]
    base_dir = run_dir or os.path.join(project_root, "data", op_dir)
    nfm_dir = os.path.join(base_dir, "nfm")
    out_dir = os.path.join(base_dir, "circulant_fourier")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "nfm_circulant_fourier.csv")

    if not os.path.isdir(nfm_dir):
        print(f"[SKIP] {nfm_dir} not found")
        return None

    pattern = re.compile(r"nfm_(.+)_step_(\d+)\.npy")
    files = []
    for filename in os.listdir(nfm_dir):
        match = pattern.match(filename)
        if match:
            files.append((int(match.group(2)), match.group(1), filename))
    files.sort()

    if not files:
        print(f"[SKIP] no NFM files in {nfm_dir}")
        return None

    fields = [
        "step",
        "source",
        "component",
        "group",
        "matrix_size",
        "D_circ",
        "D_diag",
        "D_anti",
        "R_Fourier",
        "R_sum_zero",
        "R_diff_zero",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step, component, filename in files:
            mat = np.load(os.path.join(nfm_dir, filename))
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                print(f"[SKIP] non-square NFM {filename}: shape={mat.shape}")
                continue
            mat = maybe_dlog_reorder(mat, op_key, p)
            metrics = matrix_metrics(mat, eps)
            row = {
                "step": step,
                "source": "nfm",
                "component": component,
                "group": nfm_group(component),
                **metrics,
            }
            writer.writerow(format_row(row))

    print(f"[OK] {out_csv}")
    return out_csv


def format_row(row):
    formatted = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.10f}"
        else:
            formatted[key] = value
    return formatted


def group_series(data, key_col, value_col, allowed=None):
    grouped = defaultdict(lambda: {"step": [], value_col: []})
    for step, key, value in zip(data["step"], data[key_col], data[value_col]):
        if allowed is not None and key not in allowed:
            continue
        grouped[key]["step"].append(step)
        grouped[key][value_col].append(value)
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


def plot_metric(data, metric_data, cfg, symbol, title, value_col, ylabel, out_path,
                key_col="component", allowed=None):
    grouped = group_series(data, key_col, value_col, allowed=allowed)
    if not grouped:
        return

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = plot_accuracy(ax_acc, metric_data, cfg)
    ax_acc.set_xlabel("Step")
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_right = ax_acc.twinx()
    cmap = plt.cm.tab10
    for idx, (name, values) in enumerate(grouped.items()):
        order = np.argsort(values["step"])
        steps = np.array(values["step"])[order]
        ys = np.array(values[value_col])[order]
        line, = ax_right.plot(
            steps,
            ys,
            color=cmap(idx % 10),
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=str(name),
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
    fig.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[OK] {out_path}")


def aggregate_xy(values_by_step, error_mode):
    steps, means, errors = [], [], []
    for step in sorted(values_by_step):
        values = np.asarray(values_by_step[step], dtype=float)
        steps.append(step)
        means.append(float(values.mean()))
        if len(values) <= 1:
            errors.append(0.0)
        else:
            std = float(values.std(ddof=1))
            errors.append(std / np.sqrt(len(values)) if error_mode == "sem" else std)
    return np.asarray(steps), np.asarray(means), np.asarray(errors)


def aggregate_metric_series(metric_series, value_col, error_mode):
    by_step = defaultdict(list)
    for data in metric_series:
        if value_col not in data:
            continue
        for step, value in zip(data["step"], data[value_col]):
            by_step[step].append(value)
    return aggregate_xy(by_step, error_mode)


def plot_accuracy_band(ax_acc, metric_series, cfg, error_mode):
    handles = []
    if not metric_series:
        return handles
    lw = cfg["line"]["linewidth"]
    c_acc = cfg["color"]["accuracy"]
    for value_col, linestyle, label in [
        ("train_acc", cfg["line"]["train_linestyle"], "Train Acc"),
        ("test_acc", cfg["line"]["test_linestyle"], "Test Acc"),
    ]:
        steps, mean, err = aggregate_metric_series(metric_series, value_col, error_mode)
        if len(steps) == 0:
            continue
        line, = ax_acc.plot(steps, mean, color=c_acc, linestyle=linestyle, linewidth=lw, label=label)
        ax_acc.fill_between(steps, mean - err, mean + err, color=c_acc, alpha=0.14, linewidth=0)
        handles.append(line)
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    return handles


def group_multiseed_series(data_series, key_col, value_col, allowed=None):
    grouped = defaultdict(lambda: defaultdict(list))
    for data in data_series:
        for step, key, value in zip(data["step"], data[key_col], data[value_col]):
            if allowed is not None and key not in allowed:
                continue
            grouped[key][step].append(value)
    return dict(sorted(grouped.items()))


def plot_metric_multiseed(data_series, metric_series, cfg, symbol, title, value_col,
                          ylabel, out_path, error_mode, key_col="component", allowed=None):
    grouped = group_multiseed_series(data_series, key_col, value_col, allowed=allowed)
    if not grouped:
        return

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = plot_accuracy_band(ax_acc, metric_series, cfg, error_mode)
    ax_acc.set_xlabel("Step")
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_right = ax_acc.twinx()
    cmap = plt.cm.tab10
    for idx, (name, values_by_step) in enumerate(grouped.items()):
        steps, mean, err = aggregate_xy(values_by_step, error_mode)
        color = cmap(idx % 10)
        line, = ax_right.plot(
            steps,
            mean,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=str(name),
        )
        ax_right.fill_between(steps, mean - err, mean + err, color=color, alpha=0.15, linewidth=0)
        handles.append(line)

    ax_right.set_ylabel(ylabel)
    ax_acc.set_title(f"x {symbol} y mod 97 - {title} ({error_mode})")

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
    fig.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[OK] {out_path}")


def plot_outputs(op_key, project_root, cfg, source, multi_seed=False, seeds=None, error_mode="std"):
    op_dir, symbol = OPS[op_key]
    data_dir = os.path.join(project_root, "data", op_dir, "circulant_fourier")
    metric_path = os.path.join(project_root, "data", op_dir, "metric.csv")
    metric_data = read_csv(metric_path) if os.path.exists(metric_path) else None
    out_dir = os.path.join(project_root, "results", op_dir, "circulant_fourier")
    os.makedirs(out_dir, exist_ok=True)

    if multi_seed:
        seeds = DEFAULT_SEEDS if seeds is None else seeds
        metric_series = load_seed_csvs(project_root, op_dir, seeds, "metric.csv")
        if not metric_series and metric_data is not None:
            metric_series = [metric_data]

        if source in ("all", "agop"):
            agop_series = load_seed_csvs(
                project_root, op_dir, seeds, os.path.join("circulant_fourier", "agop_circulant_fourier.csv")
            )
            if agop_series:
                plot_metric_multiseed(
                    agop_series, metric_series, cfg, symbol,
                    "AGOP circular deviation", "D_circ", r"$D_{\mathrm{circ}}$",
                    os.path.join(out_dir, "agop_D_circ_multiseed.svg"), error_mode,
                )
                plot_metric_multiseed(
                    agop_series, metric_series, cfg, symbol,
                    "AGOP Fourier-line energy", "R_Fourier", r"$R_{\mathrm{Fourier}}$",
                    os.path.join(out_dir, "agop_R_Fourier_multiseed.svg"), error_mode,
                )
            else:
                print(f"[SKIP] no multi-seed AGOP circulant/Fourier CSVs for {op_dir}")

        if source in ("all", "nfm"):
            nfm_series = load_seed_csvs(
                project_root, op_dir, seeds, os.path.join("circulant_fourier", "nfm_circulant_fourier.csv")
            )
            if nfm_series:
                for group, components in NFM_GROUPS.items():
                    plot_metric_multiseed(
                        nfm_series, metric_series, cfg, symbol,
                        f"NFM {group} circular deviation", "D_circ", r"$D_{\mathrm{circ}}$",
                        os.path.join(out_dir, f"nfm_D_circ_{group}_multiseed.svg"),
                        error_mode, allowed=set(components),
                    )
                    plot_metric_multiseed(
                        nfm_series, metric_series, cfg, symbol,
                        f"NFM {group} Fourier-line energy", "R_Fourier", r"$R_{\mathrm{Fourier}}$",
                        os.path.join(out_dir, f"nfm_R_Fourier_{group}_multiseed.svg"),
                        error_mode, allowed=set(components),
                    )
            else:
                print(f"[SKIP] no multi-seed NFM circulant/Fourier CSVs for {op_dir}")
        return

    if source in ("all", "agop"):
        agop_csv = os.path.join(data_dir, "agop_circulant_fourier.csv")
        if os.path.exists(agop_csv):
            agop = read_csv(agop_csv)
            plot_metric(
                agop,
                metric_data,
                cfg,
                symbol,
                "AGOP circular deviation",
                "D_circ",
                r"$D_{\mathrm{circ}}$",
                os.path.join(out_dir, "agop_D_circ.svg"),
            )
            plot_metric(
                agop,
                metric_data,
                cfg,
                symbol,
                "AGOP Fourier-line energy",
                "R_Fourier",
                r"$R_{\mathrm{Fourier}}$",
                os.path.join(out_dir, "agop_R_Fourier.svg"),
            )

    if source in ("all", "nfm"):
        nfm_csv = os.path.join(data_dir, "nfm_circulant_fourier.csv")
        if os.path.exists(nfm_csv):
            nfm = read_csv(nfm_csv)
            for group, components in NFM_GROUPS.items():
                plot_metric(
                    nfm,
                    metric_data,
                    cfg,
                    symbol,
                    f"NFM {group} circular deviation",
                    "D_circ",
                    r"$D_{\mathrm{circ}}$",
                    os.path.join(out_dir, f"nfm_D_circ_{group}.svg"),
                    allowed=set(components),
                )
                plot_metric(
                    nfm,
                    metric_data,
                    cfg,
                    symbol,
                    f"NFM {group} Fourier-line energy",
                    "R_Fourier",
                    r"$R_{\mathrm{Fourier}}$",
                    os.path.join(out_dir, f"nfm_R_Fourier_{group}.svg"),
                    allowed=set(components),
                )


def process_operation(op_key, project_root, cfg, p, eps, source, skip_compute,
                      multi_seed=False, seeds=None, error_mode="std"):
    print(f"\n{'=' * 60}")
    print(f"Operation: {op_key}")
    print(f"{'=' * 60}")
    op_dir, _ = OPS[op_key]
    seeds = DEFAULT_SEEDS if seeds is None else seeds

    if multi_seed:
        if not skip_compute:
            found = False
            for seed, run_dir in seed_run_dirs(project_root, op_dir, seeds):
                found = True
                print(f"[seed={seed}] run_dir={run_dir}")
                if source in ("all", "agop"):
                    compute_agop(op_key, project_root, p, eps, run_dir=run_dir)
                if source in ("all", "nfm"):
                    compute_nfm(op_key, project_root, p, eps, run_dir=run_dir)
            if not found:
                print(f"[SKIP] no seed directories found for {op_dir}; expected data/{op_dir}/seed_*/ or data/seed_*/{op_dir}/")
        plot_outputs(op_key, project_root, cfg, source, multi_seed=True,
                     seeds=seeds, error_mode=error_mode)
        return

    if not skip_compute:
        if source in ("all", "agop"):
            compute_agop(op_key, project_root, p, eps)
        if source in ("all", "nfm"):
            compute_nfm(op_key, project_root, p, eps)
    plot_outputs(op_key, project_root, cfg, source)


def main():
    parser = argparse.ArgumentParser(description="Compute and plot circulant/Fourier metrics")
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--source", choices=["all", "agop", "nfm"], default="all")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--skip-compute", action="store_true",
                        help="Only plot existing CSV files.")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Compute/plot multiple seed directories. Default seeds: 0 1 2.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--error", choices=["std", "sem"], default="std",
                        help="Error band across seeds for --multi-seed plots.")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    operations = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op_key in operations:
        process_operation(op_key, project_root, cfg, args.p, args.eps,
                          args.source, args.skip_compute, args.multi_seed,
                          args.seeds, args.error)


if __name__ == "__main__":
    main()
