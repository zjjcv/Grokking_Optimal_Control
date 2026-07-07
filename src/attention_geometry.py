#!/usr/bin/env python3
"""
Attention-geometry analysis beyond the OV pathway.

For every checkpoint and attention head this script computes five [p, p]
token-space objects and their group-alignment metrics (circulant deviation
D_circ and Fourier-line energy R_Fourier, identical to the ones used in
utils/circulant_fourier_plot.py):

    ovnfm    - E W_OV^T W_OV E^T           (baseline, as in nfm.py)
    qknfm    - E W_Q^T W_K E^T             (token-only QK bilinear, as in nfm.py)
    vnfm     - E W_V^T W_V E^T             (V pathway on its own)
    qkpos_x  - (E + pos2) W_Q^T W_K (E + pos0)^T / sqrt(d_k)
               position-aware QK score: readout-token query vs x-token key
    attmap_x - empirical attention weight query(pos2) -> key(pos0), i.e. the
               softmax attention the readout token actually pays to x, as a
               function of the input pair (x, y); label-free, uses all p^2
               inputs.  attmap_y analogous (pos2 -> pos2).

For mul/div the matrices are reordered by discrete logarithm (nonzero
elements) before computing metrics, consistent with the existing pipeline.

Outputs (per run dir):
    data/{op}[/seed_{s}]/attention_geometry/{kind}_l{l}_h{h}_step_{N}.npy
    data/{op}[/seed_{s}]/attention_geometry/attention_geometry_metrics.csv

Usage:
    python src/attention_geometry.py --operation add
    python src/attention_geometry.py --operation add --multi-seed --seeds 0 1 2
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
DEFAULT_STEPS = [100, 500, 1000, 3000, 5000, 10000, 30000, 50000, 90000]
DEFAULT_SEEDS = [0, 1, 2]
EPS = 1e-12

SAVE_NPY_KINDS = {"vnfm", "qkpos_x", "attmap_x"}


# ---------- group-alignment metrics (identical to circulant_fourier_plot) ----------

def primitive_root(p):
    phi = p - 1
    factors = set()
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"No primitive root found for p={p}")


def dlog_nonzero_order(p):
    g = primitive_root(p)
    order = []
    val = 1
    for _ in range(p - 1):
        order.append(val)
        val = (val * g) % p
    return np.array(order, dtype=int)


def maybe_dlog_reorder(mat, op_key, p):
    if op_key not in ("mul", "div"):
        return mat
    order = dlog_nonzero_order(p)
    return mat[np.ix_(order, order)]


def diag_projection(mat):
    n = mat.shape[0]
    proj = np.zeros_like(mat, dtype=np.float64)
    rows = np.arange(n)
    for offset in range(n):
        cols = (rows + offset) % n
        proj[rows, cols] = mat[rows, cols].mean()
    return proj


def anti_projection(mat):
    n = mat.shape[0]
    proj = np.zeros_like(mat, dtype=np.float64)
    rows = np.arange(n)
    for total in range(n):
        cols = (total - rows) % n
        proj[rows, cols] = mat[rows, cols].mean()
    return proj


def matrix_metrics(mat):
    m = np.asarray(mat, dtype=np.float64)
    norm = np.linalg.norm(m, ord="fro")
    d_diag = np.linalg.norm(m - diag_projection(m), ord="fro") / (norm + EPS)
    d_anti = np.linalg.norm(m - anti_projection(m), ord="fro") / (norm + EPS)

    spectrum = np.fft.fft2(m)
    energy = np.abs(spectrum) ** 2
    total = float(energy.sum())
    n = m.shape[0]
    idx = np.arange(n)
    line_sum_zero = float(energy[idx, (-idx) % n].sum()) / (total + EPS)
    line_diff_zero = float(energy[idx, idx].sum()) / (total + EPS)

    return {
        "matrix_size": n,
        "D_circ": min(d_diag, d_anti),
        "D_diag": d_diag,
        "D_anti": d_anti,
        "R_Fourier": max(line_sum_zero, line_diff_zero),
        "R_sum_zero": line_sum_zero,
        "R_diff_zero": line_diff_zero,
    }


# ---------- matrix computation ----------

def weight_matrices(model, p):
    """Weight-based token-space objects for every head."""
    results = {}
    E = model.embedding.weight[:p, :].detach().cpu().float()
    pos = model.pos_encoding.detach().cpu().float()[0]  # [3, d_model]
    d_model = E.shape[1]
    num_heads = model.blocks[0].attention.num_heads
    d_k = d_model // num_heads
    scale = d_k ** 0.5

    E_q = E + pos[2]  # readout-token query input
    E_kx = E + pos[0]  # x-token key input
    for l, block in enumerate(model.blocks):
        attn = block.attention
        W_q = attn.W_q.weight.detach().cpu().float()
        W_k = attn.W_k.weight.detach().cpu().float()
        W_v = attn.W_v.weight.detach().cpu().float()
        W_o = attn.W_o.weight.detach().cpu().float()

        for h in range(num_heads):
            sl = slice(h * d_k, (h + 1) * d_k)
            W_Qh, W_Kh, W_Vh = W_q[sl, :], W_k[sl, :], W_v[sl, :]
            W_OV = W_o[:, sl] @ W_Vh

            results[("ovnfm", l, h)] = (E @ W_OV.T @ W_OV @ E.T).numpy()
            results[("qknfm", l, h)] = (E @ W_Qh.T @ W_Kh @ E.T).numpy()
            results[("vnfm", l, h)] = (E @ W_Vh.T @ W_Vh @ E.T).numpy()
            results[("qkpos_x", l, h)] = (E_q @ W_Qh.T @ W_Kh @ E_kx.T / scale).numpy()
    return results


@torch.no_grad()
def attention_maps(model, p, device, batch_size=4096):
    """Empirical attention maps: readout-token attention to x and y positions.

    Returns dict ("attmap_x"/"attmap_y", layer, head) -> [p, p] array indexed
    by (x, y).  Label-free: all p^2 inputs are used for every operation.
    """
    num_heads = model.blocks[0].attention.num_heads
    d_k = model.blocks[0].attention.d_k
    num_layers = len(model.blocks)

    pairs = [(x, y) for x in range(p) for y in range(p)]
    inputs_all = torch.tensor([[x, p, y] for x, y in pairs], dtype=torch.long)

    maps = {("attmap_x", l, h): np.zeros((p, p)) for l in range(num_layers) for h in range(num_heads)}
    maps.update({("attmap_y", l, h): np.zeros((p, p)) for l in range(num_layers) for h in range(num_heads)})

    for start in range(0, inputs_all.shape[0], batch_size):
        batch_pairs = pairs[start:start + batch_size]
        inputs = inputs_all[start:start + batch_size].to(device)
        x = model.embedding(inputs) + model.pos_encoding[:, :inputs.shape[1], :]

        for l, block in enumerate(model.blocks):
            attn = block.attention
            B, seq_len, _ = x.shape
            Q = attn.W_q(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)
            K = attn.W_k(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)
            V = attn.W_v(x).view(B, seq_len, num_heads, d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)  # [B, heads, seq, seq]

            w_x = attn_weights[:, :, 2, 0].detach().cpu().numpy()  # readout -> x
            w_y = attn_weights[:, :, 2, 2].detach().cpu().numpy()  # readout -> y
            xs = np.fromiter((pv[0] for pv in batch_pairs), dtype=int)
            ys = np.fromiter((pv[1] for pv in batch_pairs), dtype=int)
            for h in range(num_heads):
                maps[("attmap_x", l, h)][xs, ys] = w_x[:, h]
                maps[("attmap_y", l, h)][xs, ys] = w_y[:, h]

            head_out = torch.matmul(attn_weights, V)
            head_out = head_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
            x = block.norm1(x + attn.W_o(head_out))
            x = block.norm2(x + block.ffn(x))
    return maps


# ---------- driver ----------

def resolve_seed_run_dir(project_root, op_dir, seed, multi_seed):
    data_root = os.path.join(project_root, "data")
    legacy_dir = os.path.join(data_root, op_dir)
    candidates = []
    if multi_seed:
        candidates.extend([
            os.path.join(legacy_dir, f"seed_{seed}"),
            os.path.join(data_root, f"seed_{seed}", op_dir),
        ])
    candidates.append(legacy_dir)
    for run_dir in candidates:
        if os.path.isdir(os.path.join(run_dir, "checkpoints")):
            if multi_seed and run_dir == legacy_dir:
                print(f"[WARN] seed={seed}: using legacy checkpoints without seed directory: {run_dir}")
            return run_dir
    return candidates[0]


def process_seed(args, config, project_root, op_dir, seed, device):
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_dir = os.path.join(run_dir, "attention_geometry")
    if not os.path.isdir(ckpt_dir):
        print(f"[SKIP] seed={seed}: {ckpt_dir} not found")
        return
    os.makedirs(out_dir, exist_ok=True)

    steps = args.steps or DEFAULT_STEPS
    p = config.p

    csv_path = os.path.join(out_dir, "attention_geometry_metrics.csv")
    fields = ["step", "seed", "kind", "component", "layer", "head_idx",
              "matrix_size", "D_circ", "D_diag", "D_anti",
              "R_Fourier", "R_sum_zero", "R_diff_zero"]

    print(f"seed={seed}  run_dir={run_dir}  steps={steps}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for idx, step in enumerate(steps):
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] seed={seed} step={step}: checkpoint not found")
                continue

            state = torch.load(ckpt_path, map_location="cpu")
            sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
            model = GrokkingTransformer(config).to(device)
            model.load_state_dict(sd)
            model.eval()

            objects = weight_matrices(model, p)
            objects.update(attention_maps(model, p, device))

            for (kind, l, h), mat in objects.items():
                if kind in SAVE_NPY_KINDS:
                    np.save(os.path.join(out_dir, f"{kind}_l{l}_h{h}_step_{step}.npy"), mat)
                metrics = matrix_metrics(maybe_dlog_reorder(mat, args.operation, p))
                row = {"step": step, "seed": seed, "kind": kind,
                       "component": f"l{l}_h{h}", "layer": l, "head_idx": h}
                row.update({k: (f"{v:.10f}" if isinstance(v, float) else v)
                            for k, v in metrics.items()})
                writer.writerow(row)
            f.flush()

            att_x = [m for (k, _, _), m in objects.items() if k == "attmap_x"]
            print(f"  [{idx + 1}/{len(steps)}] seed={seed} step={step:6d} | "
                  f"objects={len(objects)}  mean attn(readout->x)={np.mean(att_x):.4f}")

    print(f"Metrics saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Attention geometry beyond the OV pathway")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help=f"Checkpoint steps. Default: {DEFAULT_STEPS}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    seeds = args.seeds if args.multi_seed else [args.seed]

    for seed in seeds:
        process_seed(args, config, project_root, op_dir, seed, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
