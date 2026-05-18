#!/usr/bin/env python3
"""
绘制 KC-BDM 和训练/测试准确率随训练步数的变化

子图 1: 训练/测试准确率
子图 2: 8 个注意力头的 KC
子图 3: FFN 层 + 输入/输出 embedding 的 KC
"""

import os
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def read_csv(path):
    """读取 CSV 文件，返回 {列名: [值]} 字典"""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        data = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                data[c].append(float(row[c]))
    return data


def smooth(y, window=51, polyorder=3):
    """Savitzky-Golay 滤波平滑"""
    if len(y) < window:
        window = len(y) if len(y) % 2 == 1 else len(y) - 1
    return savgol_filter(y, window, polyorder)


def main():
    parser = argparse.ArgumentParser(description='绘制 KC-BDM 分析图')
    parser.add_argument('--operation', type=str, default='add',
                        choices=['add', 'sub', 'mul', 'div'])
    parser.add_argument('--smooth-window', type=int, default=51,
                        help='Savitzky-Golay 窗口大小')
    args = parser.parse_args()

    op_names = {'add': 'x+y', 'sub': 'x-y', 'mul': 'x*y', 'div': 'x_div_y'}
    op_symbols = {'add': '+', 'sub': '-', 'mul': '×', 'div': '÷'}
    op_dir = op_names[args.operation]
    symbol = op_symbols[args.operation]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', op_dir)

    kc_data = read_csv(os.path.join(data_dir, 'kc_bdm.csv'))
    metric_data = read_csv(os.path.join(data_dir, 'metric.csv'))

    w = args.smooth_window

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    # ---- 子图 1: 训练/测试准确率 ----
    ax = axes[0]
    ax.plot(metric_data['step'], smooth(metric_data['train_acc'], w),
            label='Train Acc', color='tab:blue', linewidth=1.5)
    ax.plot(metric_data['step'], smooth(metric_data['test_acc'], w),
            label='Test Acc', color='tab:red', linewidth=1.5)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'x {symbol} y mod 97 — Train/Test Accuracy')
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # ---- 子图 2: 8 个注意力头 KC ----
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for layer in range(2):
        for head in range(4):
            key = f'head_{layer}_{head}'
            ax.plot(kc_data['step'], smooth(kc_data[key], w),
                    label=f'L{layer}H{head}', color=colors[layer * 4 + head],
                    linewidth=1.2)
    ax.set_ylabel('KC (BDM)')
    ax.set_title('Attention Heads — Kolmogorov Complexity')
    ax.legend(ncol=4, loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 子图 3: FFN + Embedding KC ----
    ax = axes[2]
    ax.plot(kc_data['step'], smooth(kc_data['ffn_layer0'], w),
            label='FFN Layer 0', linewidth=1.2)
    ax.plot(kc_data['step'], smooth(kc_data['ffn_layer1'], w),
            label='FFN Layer 1', linewidth=1.2)
    ax.plot(kc_data['step'], smooth(kc_data['input_emb'], w),
            label='Input Embedding', linewidth=1.2)
    ax.plot(kc_data['step'], smooth(kc_data['output_emb'], w),
            label='Output Embedding', linewidth=1.2)
    ax.set_ylabel('KC (BDM)')
    ax.set_xlabel('Training Step')
    ax.set_title('FFN & Embeddings — Kolmogorov Complexity')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Grokking: x {symbol} y mod 97 — KC-BDM Analysis', fontsize=14, y=1.01)
    plt.tight_layout()

    output_path = os.path.join(data_dir, 'kc_bdm_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
