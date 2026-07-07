#!/usr/bin/env python3
"""
Train Grokking Transformers for modular arithmetic.

Legacy single-seed output:
    data/{op}/checkpoints/
    data/{op}/metric.csv
    data/{op}/train_data.json
    data/{op}/test_data.json

Seed-directory output, used by multi-seed LLC and other analyses:
    data/{op}/seed_{seed}/checkpoints/
    data/{op}/seed_{seed}/metric.csv
    data/{op}/seed_{seed}/train_data.json
    data/{op}/seed_{seed}/test_data.json
"""

import argparse
import csv
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}


class Config:
    p = 97
    num_layers = 2
    hidden_dim = 128
    num_heads = 4
    attention_dim = 128
    ffn_dim = 512
    max_len = 3

    batch_size = 512
    lr = 1e-3
    weight_decay = 0.005
    betas = (0.9, 0.98)
    total_steps = 100000
    warmup_steps = 2000

    save_interval = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_scale = 1.0
    train_ratio = 0.5

    def __init__(self, operation, seed=42, seed_dir=False, run_dir=None, overrides=None):
        self.operation = operation
        self.seed = seed
        for key, value in (overrides or {}).items():
            if not hasattr(type(self), key):
                raise ValueError(f"Unknown config override: {key}")
            setattr(self, key, value)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        op_dir = OP_DIR.get(operation, operation)
        if run_dir is not None:
            self.run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(project_root, run_dir)
        elif seed_dir:
            self.run_dir = os.path.join(project_root, "data", op_dir, f"seed_{seed}")
        else:
            self.run_dir = os.path.join(project_root, "data", op_dir)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.metric_file = os.path.join(self.run_dir, "metric.csv")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModuloDataset(Dataset):
    def __init__(self, p, operation="add", train=True, train_ratio=0.5, seed=0):
        self.p = p
        self.operation = operation
        all_pairs = [(x, y) for x in range(p) for y in range(p)]
        rng = random.Random(seed)
        rng.shuffle(all_pairs)
        n_train = int(len(all_pairs) * train_ratio)
        self.pairs = all_pairs[:n_train] if train else all_pairs[n_train:]
        print(f"[{'Train' if train else 'Test'}] Dataset size: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        if self.operation == "add":
            label = (x + y) % self.p
        elif self.operation == "sub":
            label = (x - y) % self.p
        elif self.operation == "mul":
            label = (x * y) % self.p
        elif self.operation == "div":
            label = 0 if y == 0 else (x * pow(y, -1, self.p)) % self.p
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
        input_seq = torch.tensor([x, self.p, y], dtype=torch.long)
        return input_seq, label


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.p = config.p
        self.vocab_size = config.p + 1
        self.embedding = nn.Embedding(self.vocab_size, config.attention_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.max_len, config.attention_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.attention_dim,
                config.num_heads,
                config.ffn_dim,
                dropout=0.1,
            )
            for _ in range(config.num_layers)
        ])
        self.output = nn.Linear(config.attention_dim, config.p)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = x[:, -1, :]
        return self.output(x)


class WarmupConstantScheduler:
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            lr = self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return total_loss / total, correct / total


def train(operation, seed=42, seed_dir=False, run_dir=None, overrides=None):
    set_seed(seed)
    config = Config(operation, seed=seed, seed_dir=seed_dir,
                    run_dir=run_dir, overrides=overrides)
    device = torch.device(config.device)
    symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[operation]

    print("=" * 60)
    print(f"Grokking Reproduction: x {symbol} y (mod {config.p})")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"Init Scale: {config.init_scale}")
    print(f"Train Ratio: {config.train_ratio}")
    print(f"Layers: {config.num_layers} | d_model: {config.attention_dim} | "
          f"ffn: {config.ffn_dim} | heads: {config.num_heads}")
    print(f"Total Steps: {config.total_steps}")
    print(f"Checkpoint Dir: {config.checkpoint_dir}")
    print("=" * 60)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.metric_file), exist_ok=True)

    config_snapshot = {
        "operation": operation, "seed": seed,
        "p": config.p, "num_layers": config.num_layers,
        "attention_dim": config.attention_dim, "ffn_dim": config.ffn_dim,
        "num_heads": config.num_heads, "max_len": config.max_len,
        "batch_size": config.batch_size, "lr": config.lr,
        "weight_decay": config.weight_decay, "init_scale": config.init_scale,
        "train_ratio": config.train_ratio, "total_steps": config.total_steps,
        "warmup_steps": config.warmup_steps,
    }
    with open(os.path.join(config.run_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    csv_file = open(config.metric_file, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_loss", "train_acc", "test_loss", "test_acc"])
    csv_file.flush()

    train_dataset = ModuloDataset(config.p, config.operation, train=True,
                                  train_ratio=config.train_ratio, seed=seed)
    test_dataset = ModuloDataset(config.p, config.operation, train=False,
                                 train_ratio=config.train_ratio, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    train_data_path = os.path.join(config.run_dir, "train_data.json")
    test_data_path = os.path.join(config.run_dir, "test_data.json")
    with open(train_data_path, "w") as f:
        json.dump(train_dataset.pairs, f)
    with open(test_data_path, "w") as f:
        json.dump(test_dataset.pairs, f)
    print(f"Train split saved: {train_data_path} ({len(train_dataset.pairs)} samples)")
    print(f"Test split saved: {test_data_path} ({len(test_dataset.pairs)} samples)")

    model = GrokkingTransformer(config).to(device)
    if config.init_scale != 1.0:
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(config.init_scale)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scheduler = WarmupConstantScheduler(optimizer, config.warmup_steps)
    criterion = nn.CrossEntropyLoss()

    print("\nStart training...\n")
    step = 0
    model.train()
    for _ in range((config.total_steps // len(train_loader)) + 1):
        for inputs, labels in train_loader:
            if step >= config.total_steps:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % config.save_interval == 0:
                train_loss, train_acc = evaluate(model, train_loader, criterion, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                csv_writer.writerow([
                    step,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{test_loss:.6f}",
                    f"{test_acc:.6f}",
                ])
                csv_file.flush()

                lr = scheduler.base_lr * (step / config.warmup_steps if step < config.warmup_steps else 1)
                print(
                    f"Step {step:6d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                    f"LR: {lr:.6f}"
                )

                checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    "step": step,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }, checkpoint_path)
            step += 1
        if step >= config.total_steps:
            break

    print("\nTraining complete. Final evaluation:")
    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    csv_file.close()

    final_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "seed": seed,
    }, final_path)
    print(f"\nFinal model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Grokking Transformer for modular arithmetic")
    parser.add_argument("--operation", type=str, default="add",
                        choices=["add", "sub", "mul", "div"])
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for single-seed training.")
    parser.add_argument("--seed-dir", action="store_true",
                        help="Save under data/{op}/seed_{seed}/ instead of legacy data/{op}/.")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Train multiple seeds sequentially under data/{op}/seed_{seed}/.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seeds used with --multi-seed.")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Explicit output dir (relative to project root or absolute). "
                             "Overrides the default data/{op}[/seed_{seed}] layout. "
                             "Not compatible with --multi-seed.")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--init-scale", type=float, default=None,
                        help="Multiply all initial parameters by this factor.")
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="Fraction of the p^2 pairs used for training.")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--attention-dim", type=int, default=None,
                        help="Model width d_model (must be divisible by num heads).")
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    for arg_name, cfg_name in [
        ("weight_decay", "weight_decay"), ("init_scale", "init_scale"),
        ("train_ratio", "train_ratio"), ("num_layers", "num_layers"),
        ("attention_dim", "attention_dim"), ("ffn_dim", "ffn_dim"),
        ("num_heads", "num_heads"), ("total_steps", "total_steps"),
        ("save_interval", "save_interval"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[cfg_name] = value

    if args.multi_seed:
        if args.run_dir is not None:
            raise SystemExit("--run-dir is not compatible with --multi-seed")
        for seed in args.seeds:
            train(args.operation, seed=seed, seed_dir=True, overrides=overrides)
    else:
        train(args.operation, seed=args.seed, seed_dir=args.seed_dir,
              run_dir=args.run_dir, overrides=overrides)


if __name__ == "__main__":
    main()
