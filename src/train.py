#!/usr/bin/env python3
"""
复现 Grokking 论文中的模运算实验
论文: "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"

支持四种模运算：x+y, x-y, x*y, x÷y (mod p)

使用方法:
    python train.py --operation add      # x + y mod 97
    python train.py --operation sub      # x - y mod 97
    python train.py --operation mul      # x * y mod 97
    python train.py --operation div      # x ÷ y mod 97
"""

import os
import csv
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math


# ==================== 配置参数 ====================
class Config:
    # 数据参数
    p = 97  # 模数

    # 模型参数 (与原论文一致)
    num_layers = 2
    hidden_dim = 128
    num_heads = 4
    attention_dim = 128
    ffn_dim = 512
    max_len = 3  # 输入序列长度: [x, op, y]

    # 训练参数 (原论文设置)
    batch_size = 512
    lr = 1e-3
    weight_decay = 0.005
    betas = (0.9, 0.98)
    total_steps = 100000  # 总训练步数
    warmup_steps = 2000

    # 保存和记录
    save_interval = 100  # 每100步保存
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, operation):
        self.operation = operation
        # 根据运算类型设置路径
        op_names = {
            'add': 'x+y',
            'sub': 'x-y',
            'mul': 'x*y',
            'div': 'x_div_y'
        }
        op_dir = op_names.get(operation, operation)
        # 获取项目根目录（src/ 的上级）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.checkpoint_dir = os.path.join(project_root, 'data', op_dir, 'checkpoints')
        self.metric_file = os.path.join(project_root, 'data', op_dir, 'metric.csv')


# ==================== 数据集 ====================
class ModuloDataset(Dataset):
    """模运算数据集"""

    def __init__(self, p, operation="add", train=True, train_ratio=0.5, seed=0):
        self.p = p
        self.operation = operation

        # 生成所有可能的 (x, y) 组合
        all_pairs = []
        for x in range(p):
            for y in range(p):
                all_pairs.append((x, y))

        # 使用固定随机种子进行随机划分
        import random
        random.seed(seed)
        random.shuffle(all_pairs)

        # 划分训练集和测试集
        n_train = int(len(all_pairs) * train_ratio)
        if train:
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]

        print(f"[{'Train' if train else 'Test'}] Dataset size: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]

        # 计算运算结果
        if self.operation == "add":
            label = (x + y) % self.p
        elif self.operation == "sub":
            label = (x - y) % self.p
        elif self.operation == "mul":
            label = (x * y) % self.p
        elif self.operation == "div":
            # 除法：x / y = x * y^(-1) mod p
            # 对于 y=0，特殊处理
            if y == 0:
                label = 0  # 或其他默认值
            else:
                # 计算模逆元
                inv_y = pow(y, -1, self.p)  # Python 3.8+
                label = (x * inv_y) % self.p
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        # 输入序列: [x, op, y]，op 映射为 p (作为特殊 token)
        input_seq = torch.tensor([x, self.p, y], dtype=torch.long)

        return input_seq, label


# ==================== 模型 ====================
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

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

        # Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        # Output
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    """Transformer 编码器块"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class GrokkingTransformer(nn.Module):
    """用于模运算的 Transformer 模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.p = config.p

        # Token embedding (0 到 p-1 是数字，p 是操作符)
        self.vocab_size = config.p + 1
        self.embedding = nn.Embedding(self.vocab_size, config.attention_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, config.max_len, config.attention_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.attention_dim,
                config.num_heads,
                config.ffn_dim,
                dropout=0.1
            )
            for _ in range(config.num_layers)
        ])

        # Output head (映射到 p 个类)
        self.output = nn.Linear(config.attention_dim, config.p)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size = x.shape[0]

        # Embedding
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 取最后一个 token 的输出
        x = x[:, -1, :]

        # Output
        logits = self.output(x)

        return logits


# ==================== 学习率调度器 ====================
class WarmupConstantScheduler:
    """预热 + 恒定学习率"""

    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


# ==================== 训练函数 ====================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
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


def train(operation):
    """训练主函数"""
    config = Config(operation)
    device = torch.device(config.device)

    op_symbols = {'add': '+', 'sub': '-', 'mul': '×', 'div': '÷'}
    symbol = op_symbols.get(operation, operation)

    print("=" * 60)
    print(f"Grokking Reproduction: x {symbol} y (mod {config.p})")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"Total Steps: {config.total_steps}")
    print(f"Checkpoint Dir: {config.checkpoint_dir}")
    print("=" * 60)

    # 创建数据目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.metric_file), exist_ok=True)

    # 初始化 CSV 文件
    csv_file = open(config.metric_file, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
    csv_file.flush()

    # 创建数据集
    train_dataset = ModuloDataset(config.p, config.operation, train=True, seed=42)
    test_dataset = ModuloDataset(config.p, config.operation, train=False, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 保存初始数据划分
    data_dir = os.path.dirname(config.metric_file)
    train_data_path = os.path.join(data_dir, 'train_data.json')
    test_data_path = os.path.join(data_dir, 'test_data.json')
    with open(train_data_path, 'w') as f:
        json.dump(train_dataset.pairs, f)
    with open(test_data_path, 'w') as f:
        json.dump(test_dataset.pairs, f)
    print(f"Train split saved: {train_data_path} ({len(train_dataset.pairs)} samples)")
    print(f"Test split saved: {test_data_path} ({len(test_dataset.pairs)} samples)")

    # 创建模型
    model = GrokkingTransformer(config).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = WarmupConstantScheduler(optimizer, config.warmup_steps)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    print("\n开始训练...\n")

    # 训练循环
    step = 0
    model.train()

    for epoch in range((config.total_steps // len(train_loader)) + 1):
        for inputs, labels in train_loader:
            if step >= config.total_steps:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 每100步评估和保存
            if step % config.save_interval == 0:
                train_loss, train_acc = evaluate(model, train_loader, criterion, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)

                csv_writer.writerow([step, f"{train_loss:.6f}", f"{train_acc:.6f}",
                                     f"{test_loss:.6f}", f"{test_acc:.6f}"])
                csv_file.flush()

                lr = scheduler.base_lr * (step / config.warmup_steps if step < config.warmup_steps else 1)

                print(f"Step {step:6d} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                      f"LR: {lr:.6f}")

                # 保存 checkpoint
                checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                }, checkpoint_path)

            step += 1

        if step >= config.total_steps:
            break

    # 最终评估
    print("\n训练完成！最终评估：")
    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    csv_file.close()

    # 保存最终模型
    final_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)
    print(f"\n最终模型已保存至: {final_path}")


# ==================== 主入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练 Grokking 模型进行模运算')
    parser.add_argument('--operation', type=str, default='add',
                        choices=['add', 'sub', 'mul', 'div'],
                        help='运算类型: add(加法), sub(减法), mul(乘法), div(除法)')

    args = parser.parse_args()

    train(args.operation)
