#!/usr/bin/env python3
"""
Activation patching (causal tracing) for attention heads.

Interchange interventions between clean and corrupted runs; unlike mean
ablation, patched activations always come from real forward passes, so there
is no off-distribution shift.

For every input [x, op, y] a corrupted twin [x', op, y] is built (x' random,
x' != x).  Two directions per head:
    - denoise: run corrupted input, patch in the head's clean activation.
               recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)
    - noise:   run clean input, patch in the head's corrupted activation.
               damage   = (LD_clean - LD_patched) / (LD_clean - LD_corrupt)
where LD = logit[clean answer] - logit[corrupt answer] (mean over samples).
recovery/damage = 1 means the head fully carries the causal signal, 0 means
no causal contribution.

Outputs:
    data/{op}/cma/patching.csv  - per-head, per-step, both directions

Usage:
    python src/activation_patching.py --operation add
    python src/activation_patching.py --operation mul --steps 50000 90000
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from train import Config, GrokkingTransformer

OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}


def build_pairs(p, operation, seed):
    """All valid (x, y) pairs with corrupted twins and both answers."""
    rng = np.random.default_rng(seed)
    xs, ys, xs_cor, ans_clean, ans_cor = [], [], [], [], []
    for x in range(p):
        for y in range(p):
            if operation == "div" and y == 0:
                continue
            x_cor = int(rng.integers(0, p - 1))
            if x_cor >= x:
                x_cor += 1  # uniform over values != x
            if operation == "add":
                a_c, a_k = (x + y) % p, (x_cor + y) % p
            elif operation == "sub":
                a_c, a_k = (x - y) % p, (x_cor - y) % p
            elif operation == "mul":
                a_c, a_k = (x * y) % p, (x_cor * y) % p
            elif operation == "div":
                inv = pow(y, -1, p)
                a_c, a_k = (x * inv) % p, (x_cor * inv) % p
            if a_c == a_k:
                continue  # logit diff undefined when both answers coincide
            xs.append(x)
            ys.append(y)
            xs_cor.append(x_cor)
            ans_clean.append(a_c)
            ans_cor.append(a_k)

    clean = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    corrupt = torch.tensor([[x, p, y] for x, y in zip(xs_cor, ys)], dtype=torch.long)
    return clean, corrupt, torch.tensor(ans_clean), torch.tensor(ans_cor)


@torch.no_grad()
def forward_cache(model, inputs, patch_head=None, patch_cache=None):
    """Manual forward; returns (logits, cache) with per-head attn*V outputs.

    When patch_head=(layer, head) is set, that head's output is replaced by
    patch_cache[layer][:, head] (a cache from another run on paired inputs).
    """
    num_heads = model.blocks[0].attention.num_heads
    d_k = model.blocks[0].attention.d_k

    x = model.embedding(inputs) + model.pos_encoding[:, :inputs.shape[1], :]
    cache = []

    for l, block in enumerate(model.blocks):
        attn = block.attention
        B, seq_len, _ = x.shape

        Q = attn.W_q(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)
        K = attn.W_k(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)
        V = attn.W_v(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        head_out = torch.matmul(attn_weights, V)  # [B, heads, seq, d_k]

        if patch_head is not None and l == patch_head[0]:
            head_out[:, patch_head[1], :, :] = patch_cache[l][:, patch_head[1], :, :]
        cache.append(head_out)

        head_out_m = head_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        x = block.norm1(x + attn.W_o(head_out_m))
        x = block.norm2(x + block.ffn(x))

    logits = model.output(x[:, -1, :])
    return logits, cache


def mean_logit_diff(logits, ans_clean, ans_cor):
    idx = torch.arange(logits.shape[0], device=logits.device)
    return (logits[idx, ans_clean] - logits[idx, ans_cor]).mean().item()


@torch.no_grad()
def patch_all_heads(model, clean, corrupt, ans_clean, ans_cor, device, batch_size=2048):
    """Returns clean/corrupt logit diffs and per-head patched logit diffs."""
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads
    heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]

    sums = {
        "clean": 0.0,
        "corrupt": 0.0,
        **{("denoise", lh): 0.0 for lh in heads},
        **{("noise", lh): 0.0 for lh in heads},
    }
    n_total = clean.shape[0]

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        cl = clean[start:end].to(device)
        co = corrupt[start:end].to(device)
        ac = ans_clean[start:end].to(device)
        ak = ans_cor[start:end].to(device)
        n = end - start

        logits_clean, cache_clean = forward_cache(model, cl)
        logits_cor, cache_cor = forward_cache(model, co)
        sums["clean"] += mean_logit_diff(logits_clean, ac, ak) * n
        sums["corrupt"] += mean_logit_diff(logits_cor, ac, ak) * n

        for lh in heads:
            # denoise: corrupted run, clean activation patched in
            logits_p, _ = forward_cache(model, co, patch_head=lh, patch_cache=cache_clean)
            sums[("denoise", lh)] += mean_logit_diff(logits_p, ac, ak) * n
            # noise: clean run, corrupted activation patched in
            logits_p, _ = forward_cache(model, cl, patch_head=lh, patch_cache=cache_cor)
            sums[("noise", lh)] += mean_logit_diff(logits_p, ac, ak) * n

    ld_clean = sums["clean"] / n_total
    ld_cor = sums["corrupt"] / n_total
    denom = ld_clean - ld_cor

    rows = []
    for l, h in heads:
        ld_den = sums[("denoise", (l, h))] / n_total
        ld_noi = sums[("noise", (l, h))] / n_total
        recovery = (ld_den - ld_cor) / denom if abs(denom) > 1e-8 else float("nan")
        damage = (ld_clean - ld_noi) / denom if abs(denom) > 1e-8 else float("nan")
        rows.append({
            "head": f"l{l}_h{h}", "layer": l, "head_idx": h,
            "ld_clean": ld_clean, "ld_corrupt": ld_cor,
            "ld_denoise": ld_den, "ld_noise": ld_noi,
            "recovery": recovery, "damage": damage,
        })
    return rows


def main():
    DEFAULT_STEPS = [100, 1000, 3000, 5000, 10000, 30000, 50000, 90000]

    parser = argparse.ArgumentParser(description="Activation patching / causal tracing")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help=f"Checkpoint steps. Default: {DEFAULT_STEPS}")
    parser.add_argument("--seed", type=int, default=0, help="Corruption seed.")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    steps = args.steps or DEFAULT_STEPS

    clean, corrupt, ans_clean, ans_cor = build_pairs(config.p, args.operation, args.seed)
    print(f"operation={args.operation}  device={device}  pairs={clean.shape[0]}  steps={steps}")

    model = GrokkingTransformer(config).to(device)
    model.eval()

    all_rows = []
    for idx, step in enumerate(steps):
        ckpt_path = os.path.join(project_root, "data", op_dir, "checkpoints",
                                 f"checkpoint_step_{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] checkpoint_step_{step}.pt not found")
            continue
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(sd)
        model.eval()

        rows = patch_all_heads(model, clean, corrupt, ans_clean, ans_cor,
                               device, args.batch_size)
        for r in rows:
            r["step"] = step
        all_rows.extend(rows)

        top = max(rows, key=lambda r: r["recovery"])
        print(f"[{idx + 1}/{len(steps)}] step={step:6d} | "
              f"LD_clean={rows[0]['ld_clean']:.4f} LD_corrupt={rows[0]['ld_corrupt']:.4f} | "
              f"top denoise: {top['head']} recovery={top['recovery']:.4f}")

    if not all_rows:
        print("No results to save.")
        return

    csv_dir = os.path.join(project_root, "data", op_dir, "cma")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "patching.csv")
    fields = ["step", "head", "layer", "head_idx", "ld_clean", "ld_corrupt",
              "ld_denoise", "ld_noise", "recovery", "damage"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in r.items()})
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
