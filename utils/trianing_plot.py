#!/usr/bin/env python3
"""
绘制四种模运算的训练/测试准确率和损失曲线（单图双 y 轴），保存为 PDF。

数据来源: data/{x+y,x-y,x*y,x_div_y}/metric.csv
输出目录: results/{x+y,x-y,x*y,x_div_y}/

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
    csv_path = os.path.join(data_dir, "metric.csv")

    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found")
        return

    data = read_csv(csv_path)
    out_dir = os.path.join(project_root, "results", op_dir)
    os.makedirs(out_dir, exist_ok=True)

    lw = cfg["line"]["linewidth"]
    ls_train = cfg["line"]["train_linestyle"]
    ls_test = cfg["line"]["test_linestyle"]
    c_acc = cfg["color"]["accuracy"]
    c_loss = cfg["color"]["loss"]
    leg = cfg["legend"]

    fig, ax_acc = plt.subplots(figsize=cfg["figure"]["figsize"])
    step = data["step"]

    # 左 y 轴: Accuracy
    l1, = ax_acc.plot(step, data["train_acc"], color=c_acc, linestyle=ls_train,
                      linewidth=lw, label="Train Acc")
    l2, = ax_acc.plot(step, data["test_acc"], color=c_acc, linestyle=ls_test,
                      linewidth=lw, label="Test Acc")
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy", color=c_acc)
    ax_acc.tick_params(axis="y", labelcolor=c_acc)
    ax_acc.set_xscale(cfg["axis"]["x_scale"])
    ax_acc.set_ylim(cfg["axis"]["acc_ylim"])
    ax_acc.grid(True, alpha=cfg["grid"]["alpha"])

    # 右 y 轴: Loss
    ax_loss = ax_acc.twinx()
    l3, = ax_loss.plot(step, data["train_loss"], color=c_loss, linestyle=ls_train,
                       linewidth=lw, label="Train Loss")
    l4, = ax_loss.plot(step, data["test_loss"], color=c_loss, linestyle=ls_test,
                       linewidth=lw, label="Test Loss")
    ax_loss.set_ylabel("Loss", color=c_loss)
    ax_loss.tick_params(axis="y", labelcolor=c_loss)

    ax_acc.set_title(f"x {symbol} y mod 97")

    # 底部共享图例（figure 坐标）
    fig.legend(handles=[l1, l2, l3, l4], labels=[l.get_label() for l in [l1, l2, l3, l4]],
               loc=leg["loc"], bbox_to_anchor=tuple(leg["bbox_to_anchor"]),
               bbox_transform=fig.transFigure,
               ncol=leg["ncol"], frameon=leg["frameon"], fontsize=leg["fontsize"])

    fig.subplots_adjust(bottom=0.18)
    pdf_path = os.path.join(out_dir, "training_curves.pdf")
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
