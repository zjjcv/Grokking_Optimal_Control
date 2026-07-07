#!/usr/bin/env python3
"""
Cross-validated complexity proxies for the learned prediction function.

BDM/CTM estimates are sensitive to block choice, discretization, and
reindexing.  This script therefore computes several complementary proxies
from the model prediction table

    P_t(a, b) = argmax_c f_theta_t(a, b)_c.

For each checkpoint it collects:
  1. BDM/CTM per cell for binary encodings under multiple BDM block shapes.
  2. Compression complexity from packed binary planes using zlib and lzma.
  3. Fourier-basis sparsity from 2D FFT energy of binary planes.

For mul/div it also computes a discrete-log coordinate version over Z_p^*:
rows/cols are restricted to nonzero tokens and reordered as x=g^i, y=g^j;
nonzero outputs are mapped to log_g(z), while zero predictions use symbol p-1.

Outputs:
    data/{op}/function_complexity/function_tables_step_{step}.npz
    data/{op}/function_complexity/function_complexity_bdm.csv
    data/{op}/function_complexity/function_complexity_compression_fourier.csv

Usage:
    python src/function_complexity.py --operation add
    python src/function_complexity.py --operation all --steps 1000 10000 90000
"""

import argparse
import csv
import json
import lzma
import os
import sys
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, GrokkingTransformer

from pybdm import BDM
from pybdm.partitions import PartitionIgnore


OP_DIR = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}

DEFAULT_STEPS = [100, 500, 1000, 3000, 10000, 30000, 50000, 90000]
DEFAULT_BDM_SHAPES = ["3x3", "4x4"]


def compute_label(x, y, p, operation):
    if operation == "add":
        return (x + y) % p
    if operation == "sub":
        return (x - y) % p
    if operation == "mul":
        return (x * y) % p
    if operation == "div":
        return 0 if y == 0 else (x * pow(y, -1, p)) % p
    raise ValueError(f"Unknown operation: {operation}")


def make_full_loader(p, operation, batch_size):
    xs, ys, labels = [], [], []
    for x in range(p):
        for y in range(p):
            xs.append(x)
            ys.append(y)
            labels.append(compute_label(x, y, p, operation))

    inputs = torch.tensor([[x, p, y] for x, y in zip(xs, ys)], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, labels_t), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def prediction_table(model, loader, p, device):
    model.eval()
    preds, labels = [], []
    for inputs, label_batch in loader:
        logits = model(inputs.to(device))
        preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
        labels.append(label_batch.numpy())
    p_table = np.concatenate(preds).reshape(p, p).astype(np.uint8)
    y_table = np.concatenate(labels).reshape(p, p).astype(np.uint8)
    return p_table, y_table


def primitive_root_mod_p(p):
    factors = []
    x = p - 1
    q = 2
    while q * q <= x:
        if x % q == 0:
            factors.append(q)
            while x % q == 0:
                x //= q
        q += 1
    if x > 1:
        factors.append(x)

    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    raise RuntimeError(f"No primitive root found for p={p}")


def dlog_maps(p):
    g = primitive_root_mod_p(p)
    order, log = [], {}
    x = 1
    for k in range(p - 1):
        order.append(x)
        log[x] = k
        x = (x * g) % p
    return np.array(order, dtype=int), log


def multiplicative_table_dlog(table, p, zero_symbol=None):
    if zero_symbol is None:
        zero_symbol = p - 1
    order, log = dlog_maps(p)
    restricted = table[np.ix_(order, order)]
    out = np.zeros_like(restricted, dtype=np.uint8)
    for i in range(restricted.shape[0]):
        for j in range(restricted.shape[1]):
            z = int(restricted[i, j])
            out[i, j] = zero_symbol if z == 0 else log[z]
    return out


def bitplane_encoding(table, num_bits):
    table = np.asarray(table, dtype=np.uint8)
    return np.stack([((table >> r) & 1).astype(np.uint8) for r in range(num_bits)], axis=0)


def onehot_encoding(table, num_classes):
    table = np.asarray(table, dtype=np.int64)
    return np.stack([(table == c).astype(np.uint8) for c in range(num_classes)], axis=0)


def binary_encoding(table):
    return np.asarray(table, dtype=np.uint8)[None, :, :]


def parse_shape(shape_str):
    a, b = shape_str.lower().split("x")
    return int(a), int(b)


def make_bdm(shape_str):
    shape = parse_shape(shape_str)
    try:
        return BDM(
            ndim=2,
            nsymbols=2,
            shape=shape,
            partition=PartitionIgnore,
            warn_if_missing_ctm=False,
            raise_if_zero=False,
        )
    except TypeError:
        return BDM(ndim=2, nsymbols=2, shape=shape, partition=PartitionIgnore)


def bdm_planes_per_cell(planes, shape_str):
    bdm = make_bdm(shape_str)
    vals, ents = [], []
    for plane in planes:
        binary = np.asarray(plane, dtype=int)
        try:
            vals.append(float(bdm.bdm(binary, normalized=False)))
        except Exception:
            vals.append(float("nan"))
        try:
            ents.append(float(bdm.ent(binary, normalized=False)))
        except TypeError:
            ents.append(float(bdm.ent(binary)))
        except Exception:
            ents.append(float("nan"))

    cells = planes.shape[1] * planes.shape[2]
    return float(np.nanmean(vals) / cells), float(np.nanmean(ents) / cells)


def compression_metrics(planes):
    packed = np.packbits(np.asarray(planes, dtype=np.uint8).reshape(-1)).tobytes()
    cells = planes.shape[1] * planes.shape[2]
    entries = planes.size
    zlib_bits = 8 * len(zlib.compress(packed, level=9))
    lzma_bits = 8 * len(lzma.compress(packed, preset=9))
    return {
        "zlib_bits_per_cell": zlib_bits / cells,
        "lzma_bits_per_cell": lzma_bits / cells,
        "zlib_bits_per_binary": zlib_bits / entries,
        "lzma_bits_per_binary": lzma_bits / entries,
    }


def fourier_metrics(planes, topk):
    entropies, effective, top1, topk_energy = [], [], [], []
    for plane in planes:
        spectrum = np.fft.fft2(np.asarray(plane, dtype=np.float64))
        energy = np.abs(spectrum) ** 2
        total = float(energy.sum())
        if total <= 1e-30:
            continue

        probs = (energy / total).reshape(-1)
        probs = probs[probs > 0]
        h = float(-np.sum(probs * np.log(probs)))
        max_h = np.log(energy.size)
        entropies.append(h / (max_h + 1e-30))
        effective.append(np.exp(h) / energy.size)

        sorted_energy = np.sort(energy.reshape(-1))[::-1]
        top1.append(float(sorted_energy[0] / total))
        k = min(topk, sorted_energy.size)
        topk_energy.append(float(sorted_energy[:k].sum() / total))

    if not entropies:
        return {
            "fourier_entropy_norm": float("nan"),
            "fourier_effective_fraction": float("nan"),
            "fourier_top1_energy": float("nan"),
            f"fourier_top{topk}_energy": float("nan"),
        }

    return {
        "fourier_entropy_norm": float(np.mean(entropies)),
        "fourier_effective_fraction": float(np.mean(effective)),
        "fourier_top1_energy": float(np.mean(top1)),
        f"fourier_top{topk}_energy": float(np.mean(topk_energy)),
    }


def format_float(value):
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.10f}"
    return value


def write_row(writer, row):
    writer.writerow({k: format_float(v) for k, v in row.items()})


def build_representations(p_table, y_table, operation, p, num_bits, include_onehot):
    reps = []

    reps.append(("natural", "pred", "bitplane", bitplane_encoding(p_table, num_bits)))
    if include_onehot:
        reps.append(("natural", "pred", "onehot", onehot_encoding(p_table, p)))

    correct = (p_table == y_table).astype(np.uint8)
    reps.append(("natural", "correct", "binary", binary_encoding(correct)))

    if operation in ("mul", "div"):
        p_log = multiplicative_table_dlog(p_table, p)
        y_log = multiplicative_table_dlog(y_table, p)
        reps.append(("dlog", "pred", "bitplane", bitplane_encoding(p_log, num_bits)))
        if include_onehot:
            reps.append(("dlog", "pred", "onehot", onehot_encoding(p_log, p)))
        correct_log = (p_log == y_log).astype(np.uint8)
        reps.append(("dlog", "correct", "binary", binary_encoding(correct_log)))

    return reps


def discover_steps(ckpt_dir):
    steps = []
    for filename in os.listdir(ckpt_dir):
        if filename.startswith("checkpoint_step_") and filename.endswith(".pt"):
            steps.append(int(filename.split("_")[-1].split(".")[0]))
    return sorted(steps)


def load_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def process_operation(args, operation):
    config = Config(operation)
    p = config.p
    device = "cuda" if torch.cuda.is_available() else "cpu"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[operation]
    ckpt_dir = os.path.join(project_root, "data", op_dir, "checkpoints")
    out_dir = os.path.join(project_root, "data", op_dir, "function_complexity")
    os.makedirs(out_dir, exist_ok=True)

    steps = args.steps or discover_steps(ckpt_dir)
    loader = make_full_loader(p, operation, args.batch_size)
    model = GrokkingTransformer(config).to(device)

    bdm_csv = os.path.join(out_dir, "function_complexity_bdm.csv")
    cf_csv = os.path.join(out_dir, "function_complexity_compression_fourier.csv")

    bdm_fields = [
        "step", "coord", "target", "encoding", "n", "n_planes", "bdm_shape",
        "bdm_per_cell", "entropy_per_cell", "accuracy",
    ]
    cf_fields = [
        "step", "coord", "target", "encoding", "n", "n_planes",
        "zlib_bits_per_cell", "lzma_bits_per_cell",
        "zlib_bits_per_binary", "lzma_bits_per_binary",
        "fourier_entropy_norm", "fourier_effective_fraction",
        "fourier_top1_energy", f"fourier_top{args.fourier_topk}_energy",
        "accuracy",
    ]

    print("=" * 60)
    print("Function complexity cross-validation")
    print(f"operation   = {operation}")
    print(f"device      = {device}")
    print(f"steps       = {steps}")
    print(f"bdm_shapes  = {args.bdm_shapes}")
    print(f"onehot      = {args.include_onehot}")
    print(f"output      = {out_dir}")
    print("=" * 60)

    with open(bdm_csv, "w", newline="") as bdm_file, open(cf_csv, "w", newline="") as cf_file:
        bdm_writer = csv.DictWriter(bdm_file, fieldnames=bdm_fields)
        cf_writer = csv.DictWriter(cf_file, fieldnames=cf_fields)
        bdm_writer.writeheader()
        cf_writer.writeheader()
        bdm_file.flush()
        cf_file.flush()

        for idx, step in enumerate(steps):
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] step {step}: checkpoint not found")
                continue

            model.load_state_dict(load_state_dict(ckpt_path))
            p_table, y_table = prediction_table(model, loader, p, device)
            accuracy = float(np.mean(p_table == y_table))

            table_path = os.path.join(out_dir, f"function_tables_step_{step}.npz")
            save_dict = {"P": p_table, "Y": y_table}
            if operation in ("mul", "div"):
                save_dict["P_dlog"] = multiplicative_table_dlog(p_table, p)
                save_dict["Y_dlog"] = multiplicative_table_dlog(y_table, p)
            np.savez_compressed(table_path, **save_dict)

            reps = build_representations(
                p_table,
                y_table,
                operation,
                p,
                args.num_bits,
                args.include_onehot,
            )

            for coord, target, encoding, planes in reps:
                base = {
                    "step": step,
                    "coord": coord,
                    "target": target,
                    "encoding": encoding,
                    "n": planes.shape[1],
                    "n_planes": planes.shape[0],
                    "accuracy": accuracy,
                }

                cf_row = {
                    **base,
                    **compression_metrics(planes),
                    **fourier_metrics(planes, args.fourier_topk),
                }
                write_row(cf_writer, cf_row)

                for shape in args.bdm_shapes:
                    bdm_per_cell, entropy_per_cell = bdm_planes_per_cell(planes, shape)
                    bdm_row = {
                        **base,
                        "bdm_shape": shape,
                        "bdm_per_cell": bdm_per_cell,
                        "entropy_per_cell": entropy_per_cell,
                    }
                    write_row(bdm_writer, bdm_row)

            bdm_file.flush()
            cf_file.flush()
            print(f"[{idx + 1}/{len(steps)}] step={step:6d} acc={accuracy:.4f}")

    print(f"Saved: {bdm_csv}")
    print(f"Saved: {cf_csv}\n")


def main():
    parser = argparse.ArgumentParser(description="Collect complexity proxies for learned functions")
    parser.add_argument("--operation", choices=list(OP_DIR.keys()) + ["all"], default="add")
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-bits", type=int, default=7)
    parser.add_argument("--bdm-shapes", nargs="+", default=list(DEFAULT_BDM_SHAPES))
    parser.add_argument("--fourier-topk", type=int, default=16)
    parser.add_argument("--include-onehot", action="store_true",
                        help="Also evaluate one-hot class-plane discretization.")
    args = parser.parse_args()

    operations = list(OP_DIR.keys()) if args.operation == "all" else [args.operation]
    for operation in operations:
        process_operation(args, operation)


if __name__ == "__main__":
    main()
