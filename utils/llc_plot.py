#!/usr/bin/env python3
"""
绘制 LLC 与 Train/Test Accuracy 随训练步数的变化（单图双 y 轴）。

左 y 轴: Accuracy (蓝色)
右 y 轴: LLC (绿色)

数据来源:
  - data/{op}/llc.csv
  - data/{op}/metric.csv
输出: results/{op}/llc_plot.pdf

绘图配置: utils/plot_config.json
"""

import os
import json
import csv
import argparse
import matplotlib.pyplot as plt


OPS = {
    "add": ("x+y", "+"),
    "sub": ("x-y", "-"),
    "mul": ("x*y", r"\times"),
    "div": ("x_div_y", r"\div"),
}


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_config.json")
    with open(config_path, "r") as f:
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


def plot_one(op_key, project_root, cfg):
    op_dir, symbol = OPS[op_key]
    data_dir = os.path.join(project_root, "data", op_dir)
    llc_path = os.path.join(data_dir, "llc.csv")
    metric_path = os.path.join(data_dir, "metric.csv")

    if not os.path.exists(llc_path):
        print(f"[SKIP] {llc_path} not found")
        return
    if not os.path.exists(metric_path):
        print(f"[SKIP] {metric_path} not found")
        return

    llc_data = read_csv(llc_path)
    metric_data = read_csv(metric_path)

    out_dir = os.path.join(project_root, "results", op_dir)
    os.makedirs(out_dir, exist_ok=True)

    lw = cfg["line"]["linewidth"]
    ls_train = cfg["line"]["train_linestyle"]
    ls_test = cfg["line"]["test_linestyle"]
    c_acc = cfg["color"]["accuracy"]
    c_llc = "tab:green"
    leg = cfg["legend"]

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])

    # 左 y 轴: Accuracy
    l1, = ax_acc.plot(metric_data["step"], metric_data["train_acc"],
                      color=c_acc, linestyle=ls_train, linewidth=lw, label="Train Acc")
    l2, = ax_acc.plot(metric_data["step"], metric_data["test_acc"],
                      color=c_acc, linestyle=ls_test, linewidth=lw, label="Test Acc")
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    # 右 y 轴: LLC
    ax_llc = ax_acc.twinx()
    l3, = ax_llc.plot(llc_data["step"], llc_data["llc_mean"],
                      color=c_llc, linewidth=lw, label="LLC")
    ax_llc.set_ylabel("LLC", color=c_llc)
    ax_llc.tick_params(axis="y", labelcolor=c_llc)

    ax_acc.set_title(f"x {symbol} y mod 97")

    # 底部共享图例
    handles = [l1, l2, l3]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure, ncol=3,
               frameon=leg["frameon"], fontsize=leg["fontsize"])

    fig.subplots_adjust(bottom=0.18)
    pdf_path = os.path.join(out_dir, "llc_plot.pdf")
    plt.savefig(pdf_path)
    plt.close()
    print(f"[OK] {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config()
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]

    for op in ops:
        plot_one(op, project_root, cfg)


if __name__ == "__main__":
    main()
