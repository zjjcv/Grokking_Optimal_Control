#!/usr/bin/env python3
"""
使用 devinterp 估算各 checkpoint 的 Local Learning Coefficient (LLC)

对 data/{op}/checkpoints/ 下的权重文件逐个计算 LLC，
输出 data/{op}/llc.csv（列: step, llc_mean, llc_std）

用法:
    python src/llc.py --operation add
    python src/llc.py --operation add --step-interval 500   # 每隔500步采样一次
"""

import os
import sys
import csv
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 将 src/ 加入 path 以便直接 import train.py 中的模型
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import GrokkingTransformer, Config

from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD


def evaluate(model, data):
    """devinterp 要求的 evaluate 签名: (model, batch) -> (loss, dict)"""
    inputs, labels = data
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)
    return loss, {}


def make_loader(p, operation, batch_size=512, seed=42):
    """构建完整训练集的 DataLoader 用于 SGLD 采样"""
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", {"add": "x+y", "sub": "x-y",
                                    "mul": "x*y", "div": "x_div_y"}[operation],
                           "train_data.json"), "r") as f:
        pairs = json.load(f)

    xs, ys, labels = [], [], []
    for x, y in pairs:
        xs.append(x)
        ys.append(y)
        if operation == "add":
            labels.append((x + y) % p)
        elif operation == "sub":
            labels.append((x - y) % p)
        elif operation == "mul":
            labels.append((x * y) % p)
        elif operation == "div":
            if y == 0:
                labels.append(0)
            else:
                labels.append((x * pow(y, -1, p)) % p)

    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(inputs, labels_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    parser = argparse.ArgumentParser(description="Compute LLC for grokking checkpoints")
    parser.add_argument("--operation", type=str, default="add",
                        choices=["add", "sub", "mul", "div"])
    parser.add_argument("--num-chains", type=int, default=5)
    parser.add_argument("--num-draws", type=int, default=100)
    parser.add_argument("--num-burnin-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--step-interval", type=int, default=0,
                        help="只处理 step 为此值倍数的 checkpoint (0=全部)")
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    op_dir = {"add": "x+y", "sub": "x-y", "mul": "x*y", "div": "x_div_y"}[args.operation]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, "data", op_dir, "checkpoints")
    output_csv = os.path.join(project_root, "data", op_dir, "llc.csv")

    # 收集并排序 checkpoints
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir)
         if f.startswith("checkpoint_step_") and f.endswith(".pt")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    if args.step_interval > 0:
        ckpt_files = [f for f in ckpt_files
                      if int(f.split("_")[-1].split(".")[0]) % args.step_interval == 0]

    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")
    print(f"device={device}  chains={args.num_chains}  draws={args.num_draws}")

    loader = make_loader(config.p, args.operation)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "llc_mean", "llc_std"])
        f.flush()

        for idx, ckpt_file in enumerate(ckpt_files):
            step = int(ckpt_file.split("_")[-1].split(".")[0])
            path = os.path.join(ckpt_dir, ckpt_file)

            state_dict = torch.load(path, map_location="cpu")["model_state_dict"]
            model = GrokkingTransformer(config).to(device)
            model.load_state_dict(state_dict)

            results = estimate_learning_coeff_with_summary(
                model=model,
                loader=loader,
                evaluate=evaluate,
                sampling_method=SGLD,
                optimizer_kwargs={
                    "lr": args.lr,
                    "noise_level": args.noise_level,
                },
                num_draws=args.num_draws,
                num_chains=args.num_chains,
                num_burnin_steps=args.num_burnin_steps,
                device=device,
                verbose=False,
            )

            llc_mean = results["llc/mean"]
            llc_std = results["llc/std"]
            writer.writerow([step, f"{llc_mean:.6f}", f"{llc_std:.6f}"])
            f.flush()

            print(f"  [{idx + 1}/{len(ckpt_files)}] Step {step:6d} | "
                  f"LLC = {llc_mean:.4f} ± {llc_std:.4f}")

    print(f"\nLLC data saved to: {output_csv}")


if __name__ == "__main__":
    main()
