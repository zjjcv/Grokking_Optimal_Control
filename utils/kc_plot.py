#!/usr/bin/env python3
"""
Plot KC (BDM) and circulant/anti-circulant alignment for weights, AGOP, and NFM.

Reads:
    data/{op}/kc.csv        — weight KC per parameter
    data/{op}/kc_agop.csv   — AGOP KC + R_circ/R_anti
    data/{op}/kc_nfm.csv    — NFM KC + R_circ/R_anti per component
    data/{op}/metric.csv    — train/test accuracy and loss

Output:
    results/{op}/kc/kc_plot.pdf              — all weight params KC
    results/{op}/kc/kc_{group}.pdf           — per-group weight KC
    results/{op}/kc/kc_agop_kc.pdf           — AGOP KC + accuracy
    results/{op}/kc/kc_agop_R_circ.pdf       — AGOP R_circ + accuracy
    results/{op}/kc/kc_agop_R_anti.pdf       — AGOP R_anti + accuracy
    results/{op}/kc/kc_nfm_kc_{group}.pdf    — NFM KC per component group
    results/{op}/kc/kc_nfm_R_circ_{group}.pdf — NFM R_circ per component group
    results/{op}/kc/kc_nfm_R_anti_{group}.pdf — NFM R_anti per component group

Usage:
    python utils/kc_plot.py --operation add
    python utils/kc_plot.py --operation all
    python utils/kc_plot.py --operation add --section agop nfm
"""

import os
import csv
import json
import argparse

import matplotlib.pyplot as plt


OPS = {
    "add": ("x+y", "+"),
    "sub": ("x-y", "-"),
    "mul": ("x_mul_y", r"\times"),
    "div": ("x_div_y", r"\div"),
}

AGOP_BLOCKS = ["full", "Gxx", "Gxy", "Gyy"]

NFM_GROUPS = {
    "embedding": ["embedding"],
    "unembedding": ["unembedding"],
    "mlp_in": ["mlp_in_l0", "mlp_in_l1"],
    "mlp_out": ["mlp_out_l0", "mlp_out_l1"],
    "qk": [f"qk_l{l}_h{h}" for l in range(2) for h in range(4)],
    "ov": [f"ov_l{l}_h{h}" for l in range(2) for h in range(4)],
}


# ==================== Helpers ====================

def load_plot_config():
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
                data[c].append(float(row[c]))
    return data


# ==================== Weight KC plots ====================

def group_params(param_names):
    """Group parameter names by component."""
    groups = {
        "embedding": [],
        "output": [],
        "pos_encoding": [],
    }
    for l in range(2):
        for part in ["attention", "ffn", "norm1", "norm2"]:
            groups[f"blocks.{l}.{part}"] = []

    for name in param_names:
        if name == "embedding.weight":
            groups["embedding"].append(name)
        elif name == "pos_encoding":
            groups["pos_encoding"].append(name)
        elif name.startswith("output."):
            groups["output"].append(name)
        elif name.startswith("blocks."):
            parts = name.split(".")
            key = f"{parts[0]}.{parts[1]}.{parts[2]}"
            if key not in groups:
                groups[key] = []
            groups[key].append(name)

    return {k: v for k, v in groups.items() if v}


def plot_overall(kc_data, metric_data, cfg, out_path, symbol):
    """Plot all parameters KC + accuracy in one figure."""
    lw = cfg["line"]["linewidth"]
    ls_train = cfg["line"]["train_linestyle"]
    ls_test = cfg["line"]["test_linestyle"]
    c_acc = cfg["color"]["accuracy"]
    leg = cfg["legend"]
    kc_color = "tab:purple"

    param_names = [c for c in kc_data.keys() if c != "step"]
    kc_steps = kc_data["step"]

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = []

    if metric_data:
        steps = metric_data["step"]
        l1, = ax_acc.plot(steps, metric_data["train_acc"], color=c_acc,
                          linestyle=ls_train, linewidth=lw, label="Train Acc")
        l2, = ax_acc.plot(steps, metric_data["test_acc"], color=c_acc,
                          linestyle=ls_test, linewidth=lw, label="Test Acc")
        handles += [l1, l2]

    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_kc = ax_acc.twinx()
    cmap = plt.cm.Purples
    n = len(param_names)
    for i, name in enumerate(param_names):
        color = cmap(0.3 + 0.6 * i / max(n - 1, 1))
        l, = ax_kc.plot(kc_steps, kc_data[name], color=color,
                        linewidth=1.2, label=name)
        handles.append(l)

    ax_kc.set_ylabel("KC (BDM)", color=kc_color)
    ax_kc.tick_params(axis="y", labelcolor=kc_color)
    ax_acc.set_title(f"x {symbol} y mod 97")

    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure,
               ncol=min(len(handles), 6), frameon=leg["frameon"],
               fontsize=leg["fontsize"])

    fig.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] {out_path}")


def plot_group(group_name, param_names, kc_data, metric_data, cfg, out_path, symbol):
    """Plot a group of parameters KC + accuracy."""
    lw = cfg["line"]["linewidth"]
    ls_train = cfg["line"]["train_linestyle"]
    ls_test = cfg["line"]["test_linestyle"]
    c_acc = cfg["color"]["accuracy"]
    leg = cfg["legend"]
    kc_color = "tab:purple"

    kc_steps = kc_data["step"]

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = []

    if metric_data:
        steps = metric_data["step"]
        l1, = ax_acc.plot(steps, metric_data["train_acc"], color=c_acc,
                          linestyle=ls_train, linewidth=lw, label="Train Acc")
        l2, = ax_acc.plot(steps, metric_data["test_acc"], color=c_acc,
                          linestyle=ls_test, linewidth=lw, label="Test Acc")
        handles += [l1, l2]

    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_kc = ax_acc.twinx()
    cmap = plt.cm.Purples
    n = len(param_names)
    for i, name in enumerate(param_names):
        color = cmap(0.3 + 0.6 * i / max(n - 1, 1))
        short = name.split(".")[-1]
        l, = ax_kc.plot(kc_steps, kc_data[name], color=color,
                        linewidth=1.8, label=short)
        handles.append(l)

    ax_kc.set_ylabel("KC (BDM)", color=kc_color)
    ax_kc.tick_params(axis="y", labelcolor=kc_color)
    ax_acc.set_title(f"x {symbol} y mod 97 — {group_name}")

    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure,
               ncol=min(len(handles), 6), frameon=leg["frameon"],
               fontsize=leg["fontsize"])

    fig.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] {out_path}")


def process_weights(op_key, project_root, cfg):
    """Plot weight KC (original functionality)."""
    op_dir, symbol = OPS[op_key]
    kc_csv = os.path.join(project_root, "data", op_dir, "kc.csv")
    metric_csv = os.path.join(project_root, "data", op_dir, "metric.csv")
    out_dir = os.path.join(project_root, "results", op_dir, "kc")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(kc_csv):
        print(f"[SKIP] {kc_csv} not found")
        return

    kc_data = read_csv(kc_csv)
    metric_data = read_csv(metric_csv) if os.path.exists(metric_csv) else None

    param_names = [c for c in kc_data.keys() if c != "step"]

    plot_overall(kc_data, metric_data, cfg,
                 os.path.join(out_dir, "kc_plot.pdf"), symbol)

    groups = group_params(param_names)
    for gname, gparams in groups.items():
        safe = gname.replace(".", "_")
        plot_group(gname, gparams, kc_data, metric_data, cfg,
                   os.path.join(out_dir, f"kc_{safe}.pdf"), symbol)


# ==================== AGOP plots ====================

def _plot_dual_y(data, metric_data, columns, ylabel_right, cfg, out_path, title,
                 acc_ylim=None, right_ylim=None):
    """Generic dual y-axis plot: left=accuracy, right=arbitrary metric columns."""
    lw = cfg["line"]["linewidth"]
    ls_train = cfg["line"]["train_linestyle"]
    ls_test = cfg["line"]["test_linestyle"]
    c_acc = cfg["color"]["accuracy"]
    leg = cfg["legend"]

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = []

    if metric_data:
        steps = metric_data["step"]
        l1, = ax_acc.plot(steps, metric_data["train_acc"], color=c_acc,
                          linestyle=ls_train, linewidth=lw, label="Train Acc")
        l2, = ax_acc.plot(steps, metric_data["test_acc"], color=c_acc,
                          linestyle=ls_test, linewidth=lw, label="Test Acc")
        handles += [l1, l2]

    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.set_ylim(acc_ylim or cfg["axis"]["acc_ylim"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_right = ax_acc.twinx()
    cmap = plt.cm.tab10
    steps = data["step"]
    for i, col in enumerate(columns):
        l, = ax_right.plot(steps, data[col], color=cmap(i % 10),
                           linewidth=1.6, label=col)
        handles.append(l)

    ax_right.set_ylabel(ylabel_right)
    if right_ylim is not None:
        ax_right.set_ylim(right_ylim)
    ax_acc.set_title(title)

    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure,
               ncol=min(len(handles), 6), frameon=leg["frameon"],
               fontsize=leg["fontsize"])

    fig.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] {out_path}")


def process_agop(op_key, project_root, cfg):
    """Plot AGOP KC and alignment."""
    op_dir, symbol = OPS[op_key]
    agop_csv = os.path.join(project_root, "data", op_dir, "kc_agop.csv")
    metric_csv = os.path.join(project_root, "data", op_dir, "metric.csv")
    out_dir = os.path.join(project_root, "results", op_dir, "kc")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(agop_csv):
        print(f"[SKIP] {agop_csv} not found")
        return

    agop_data = read_csv(agop_csv)
    metric_data = read_csv(metric_csv) if os.path.exists(metric_csv) else None

    # 1. AGOP KC
    kc_cols = [f"kc_{blk}" for blk in AGOP_BLOCKS]
    _plot_dual_y(agop_data, metric_data, kc_cols, "KC (BDM)", cfg,
                 os.path.join(out_dir, "kc_agop_kc.pdf"),
                 title=f"x {symbol} y mod 97 — AGOP KC")

    # 2. AGOP R_circ
    rcirc_cols = [f"R_circ_{blk}" for blk in AGOP_BLOCKS]
    _plot_dual_y(agop_data, metric_data, rcirc_cols, r"$R_{\mathrm{circ}}$", cfg,
                 os.path.join(out_dir, "kc_agop_R_circ.pdf"),
                 title=rf"x {symbol} y mod 97 — AGOP $R_{{\mathrm{{circ}}}}$")

    # 3. AGOP R_anti
    ranti_cols = [f"R_anti_{blk}" for blk in AGOP_BLOCKS]
    _plot_dual_y(agop_data, metric_data, ranti_cols, r"$R_{\mathrm{anti}}$", cfg,
                 os.path.join(out_dir, "kc_agop_R_anti.pdf"),
                 title=rf"x {symbol} y mod 97 — AGOP $R_{{\mathrm{{anti}}}}$")


# ==================== NFM plots ====================

def _plot_nfm_group(metric_type, group_name, components, nfm_data, metric_data,
                    cfg, out_path, symbol):
    """Plot one metric (kc / R_circ / R_anti) for a component group + accuracy."""
    lw = cfg["line"]["linewidth"]
    ls_train = cfg["line"]["train_linestyle"]
    ls_test = cfg["line"]["test_linestyle"]
    c_acc = cfg["color"]["accuracy"]
    leg = cfg["legend"]

    cols = [f"{metric_type}_{comp}" for comp in components]
    available_cols = [c for c in cols if c in nfm_data]
    if not available_cols:
        return

    is_kc = metric_type == "kc"
    ylabel = "KC (BDM)" if is_kc else (
        r"$R_{\mathrm{circ}}$" if "circ" in metric_type
        else r"$R_{\mathrm{anti}}$"
    )

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    handles = []

    if metric_data:
        steps = metric_data["step"]
        l1, = ax_acc.plot(steps, metric_data["train_acc"], color=c_acc,
                          linestyle=ls_train, linewidth=lw, label="Train Acc")
        l2, = ax_acc.plot(steps, metric_data["test_acc"], color=c_acc,
                          linestyle=ls_test, linewidth=lw, label="Test Acc")
        handles += [l1, l2]

    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    ax_right = ax_acc.twinx()
    cmap = plt.cm.tab10
    steps = nfm_data["step"]
    for i, col in enumerate(available_cols):
        short_label = col.split(metric_type + "_", 1)[-1] if not is_kc else col[3:]
        l, = ax_right.plot(steps, nfm_data[col], color=cmap(i % 10),
                           linewidth=1.6, label=short_label)
        handles.append(l)

    ax_right.set_ylabel(ylabel)

    metric_label = ylabel
    ax_acc.set_title(f"x {symbol} y mod 97 — NFM {group_name} {metric_label}")

    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure,
               ncol=min(len(handles), 6), frameon=leg["frameon"],
               fontsize=leg["fontsize"])

    fig.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] {out_path}")


def process_nfm(op_key, project_root, cfg, view_groups=None):
    """Plot NFM KC and alignment per component group."""
    op_dir, symbol = OPS[op_key]
    nfm_csv = os.path.join(project_root, "data", op_dir, "kc_nfm.csv")
    metric_csv = os.path.join(project_root, "data", op_dir, "metric.csv")
    out_dir = os.path.join(project_root, "results", op_dir, "kc")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(nfm_csv):
        print(f"[SKIP] {nfm_csv} not found")
        return

    nfm_data = read_csv(nfm_csv)
    metric_data = read_csv(metric_csv) if os.path.exists(metric_csv) else None

    groups = view_groups or list(NFM_GROUPS.keys())
    for gname in groups:
        if gname not in NFM_GROUPS:
            print(f"  [WARN] Unknown NFM group: {gname}")
            continue
        components = NFM_GROUPS[gname]
        available = [c for c in components if f"kc_{c}" in nfm_data]
        if not available:
            continue

        for metric_type in ["kc", "R_circ", "R_anti"]:
            _plot_nfm_group(metric_type, gname, available, nfm_data, metric_data,
                            cfg,
                            os.path.join(out_dir, f"kc_nfm_{metric_type}_{gname}.pdf"),
                            symbol)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Plot KC alongside accuracy")
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--section", nargs="*", default=None,
                        help="Which sections to plot: weights agop nfm. Default: all")
    parser.add_argument("--nfm-groups", nargs="*", default=None,
                        help="NFM component groups: embedding unembedding mlp_in mlp_out qk ov. Default: all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_plot_config()
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]
    sections = args.section or ["weights", "agop", "nfm"]

    for op in ops:
        print(f"\n{'=' * 50}")
        print(f"Operation: {op}")
        print(f"{'=' * 50}")
        if "weights" in sections:
            process_weights(op, project_root, cfg)
        if "agop" in sections:
            process_agop(op, project_root, cfg)
        if "nfm" in sections:
            process_nfm(op, project_root, cfg, args.nfm_groups)


if __name__ == "__main__":
    main()
