#!/usr/bin/env python3
"""
计算 Grokking 模型各部件的柯氏复杂度（使用 BDM 近似）

部件:
  - 8 个注意力头 (2 layers × 4 heads)
  - 2 个 FFN 层 (每层 FFN 整体)
  - 输入/输出 embedding

二值化方式: 固定阈值 0 (weight > 0 → 1)
直接对原始 2D 权重矩阵计算 BDM，同一组件内多个矩阵求和
"""

import os
import csv
import argparse
import numpy as np
import torch
from pybdm import BDM


def extract_weight_matrices(state_dict, component, layer=None, head=None):
    """提取指定组件的原始 2D 权重矩阵列表"""
    matrices = []
    d_k = 32  # 128 / 4 heads

    if component == 'head':
        for wname in ['W_q', 'W_k', 'W_v']:
            w = state_dict[f'blocks.{layer}.attention.{wname}.weight']
            matrices.append(w[head * d_k:(head + 1) * d_k, :].numpy())
        # W_o: 列对应各 head
        wo = state_dict[f'blocks.{layer}.attention.W_o.weight']
        matrices.append(wo[:, head * d_k:(head + 1) * d_k].numpy())

    elif component == 'ffn':
        for lname in ['linear1', 'linear2']:
            w = state_dict[f'blocks.{layer}.ffn.{lname}.weight']
            matrices.append(w.numpy())

    elif component == 'input_emb':
        matrices.append(state_dict['embedding.weight'].numpy())

    elif component == 'output_emb':
        matrices.append(state_dict['output.weight'].numpy())

    return matrices


def binarize_fixed(arr):
    """固定阈值 0 二值化"""
    return (arr > 0).astype(np.int8)


def compute_component_bdm(bdm, matrices):
    """对多个 2D 矩阵分别计算 BDM 后求和"""
    total = 0.0
    for mat in matrices:
        binary = binarize_fixed(mat)
        total += bdm.bdm(binary)
    return total


def main():
    parser = argparse.ArgumentParser(description='计算模型各部件的 KC (BDM)')
    parser.add_argument('--operation', type=str, default='add',
                        choices=['add', 'sub', 'mul', 'div'])
    args = parser.parse_args()

    op_names = {'add': 'x+y', 'sub': 'x-y', 'mul': 'x*y', 'div': 'x_div_y'}
    op_dir = op_names[args.operation]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(project_root, 'data', op_dir, 'checkpoints')
    output_csv = os.path.join(project_root, 'data', op_dir, 'kc_bdm.csv')

    # 收集所有 checkpoint 并按 step 排序
    ckpt_files = sorted(
        [f for f in os.listdir(checkpoint_dir)
         if f.startswith('checkpoint_step_') and f.endswith('.pt')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    print(f"Found {len(ckpt_files)} checkpoints in {checkpoint_dir}")

    bdm = BDM(ndim=2)

    # 定义组件列表
    components = []
    for layer in range(2):
        for head in range(4):
            components.append(('head', layer, head, f'head_{layer}_{head}'))
    for layer in range(2):
        components.append(('ffn', layer, None, f'ffn_layer{layer}'))
    components.append(('input_emb', None, None, 'input_emb'))
    components.append(('output_emb', None, None, 'output_emb'))

    header = ['step'] + [c[3] for c in components]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        for idx, ckpt_file in enumerate(ckpt_files):
            step = int(ckpt_file.split('_')[-1].split('.')[0])
            path = os.path.join(checkpoint_dir, ckpt_file)

            state_dict = torch.load(path, map_location='cpu')['model_state_dict']

            row = [step]
            for comp_type, layer, head, _ in components:
                matrices = extract_weight_matrices(state_dict, comp_type, layer, head)
                kc = compute_component_bdm(bdm, matrices)
                row.append(f"{kc:.6f}")

            writer.writerow(row)
            f.flush()

            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  [{idx + 1}/{len(ckpt_files)}] Step {step} done")

    print(f"\nKC-BDM data saved to: {output_csv}")


if __name__ == '__main__':
    main()
