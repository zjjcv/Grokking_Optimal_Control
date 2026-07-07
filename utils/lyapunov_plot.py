#!/usr/bin/env python3
"""
Plot empirical Lyapunov hidden-state diagnostics alongside accuracy.

Reads:
    data/{op}/lyapunov/lyapunov.csv
    data/{op}/metric.csv
Writes:
    results/{op}/lyapunov/lyapunov_V_tilde.svg
    results/{op}/lyapunov/lyapunov_V_hat.svg
    results/{op}/lyapunov/lyapunov_rho.svg

Usage:
    python utils/lyapunov_plot.py --operation add
    python utils/lyapunov_plot.py --operation all
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


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


def parse_float(value):
    if value == "" or value is None:
        return float("nan")
    return float(value)


def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        data = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                if c == "split":
                    data[c].append(row[c])
                else:
                    data[c].append(parse_float(row[c]))
    return data


def organize_by_layer(data, value_col):
    by_layer = defaultdict(lambda: {"step": [], value_col: []})
    for step, layer, value in zip(data["step"], data["layer"], data[value_col]):
        if math.isnan(value):
            continue
        layer_idx = int(layer)
        by_layer[layer_idx]["step"].append(step)
        by_layer[layer_idx][value_col].append(value)
    return dict(sorted(by_layer.items()))


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


def plot_metric(op_key, symbol, lyap_data, metric_data, cfg, value_col, ylabel, out_path):
    by_layer = organize_by_layer(lyap_data, value_col)
    if not by_layer:
        print(f"[SKIP] no values for {value_col}")
        return

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = plot_accuracy(ax_acc, metric_data, cfg)

    ax_acc.set_xlabel("Step")
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_right = ax_acc.twinx()
    cmap = plt.cm.tab10
    for idx, (layer, values) in enumerate(by_layer.items()):
        line, = ax_right.plot(
            values["step"],
            values[value_col],
            color=cmap(idx % 10),
            linewidth=cfg["line"]["linewidth"],
            marker="o",
            markersize=4,
            label=f"Layer {layer}",
        )
        handles.append(line)

    if value_col == "rho_next":
        ax_right.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

    ax_right.set_ylabel(ylabel)
    ax_acc.set_title(f"x {symbol} y mod 97 - Lyapunov {value_col}")

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
    plt.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[OK] {out_path}")


def process_operation(op_key, project_root, cfg):
    op_dir, symbol = OPS[op_key]
    lyap_csv = os.path.join(project_root, "data", op_dir, "lyapunov", "lyapunov.csv")
    metric_csv = os.path.join(project_root, "data", op_dir, "metric.csv")
    out_dir = os.path.join(project_root, "results", op_dir, "lyapunov")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(lyap_csv):
        print(f"[SKIP] {lyap_csv} not found")
        return

    lyap_data = read_csv(lyap_csv)
    metric_data = read_csv(metric_csv) if os.path.exists(metric_csv) else None

    plot_metric(
        op_key,
        symbol,
        lyap_data,
        metric_data,
        cfg,
        "V_tilde",
        r"$\widetilde{V}_{\ell,\tau}$",
        os.path.join(out_dir, "lyapunov_V_tilde.svg"),
    )
    plot_metric(
        op_key,
        symbol,
        lyap_data,
        metric_data,
        cfg,
        "V_hat",
        r"$\widehat{V}_{\ell,\tau}$",
        os.path.join(out_dir, "lyapunov_V_hat.svg"),
    )
    plot_metric(
        op_key,
        symbol,
        lyap_data,
        metric_data,
        cfg,
        "rho_next",
        r"$\rho_{\ell,\tau}=\widetilde{V}_{\ell+1,\tau}/\widetilde{V}_{\ell,\tau}$",
        os.path.join(out_dir, "lyapunov_rho.svg"),
    )


def main():
    parser = argparse.ArgumentParser(description="Plot empirical Lyapunov diagnostic")
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    operations = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in operations:
        process_operation(op, project_root, cfg)


if __name__ == "__main__":
    main()
