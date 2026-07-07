#!/usr/bin/env python3
"""
Mode connectivity between memorization and generalization checkpoints.

Trains a quadratic Bezier curve (Garipov et al. 2018) in weight space between
the same endpoints used by src/interpolation.py (default step 2000 -> 99000):

    theta(t) = (1-t)^2 theta_A + 2 t (1-t) theta_bend + t^2 theta_B

Only theta_bend is trainable (initialized at the midpoint).  Each curve-training
step samples t ~ U(0, 1) and minimizes plain train cross-entropy at theta(t).

Both the linear path and the trained Bezier curve are then evaluated on a
dense t-grid with two potentials:
    - plain train/test loss and accuracy
    - AdamW-style regularized train loss: loss + (wd / 2) * ||theta||^2
      (weight decay is the potential the optimizer actually feels; the L2 norm
      of memorization vs generalization solutions differs strongly)

Outputs (per run dir):
    data/{op}[/seed_{s}]/mode_connectivity.csv       (path, t, metrics)
    data/{op}/mode_connectivity_multiseed.csv        (aggregate, multi-seed)

Usage:
    python src/mode_connectivity.py --operation add
    python src/mode_connectivity.py --operation add --multi-seed --seeds 0 1 2
    python src/mode_connectivity.py --operation mul --start 2000 --end 99000
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from train import Config, GrokkingTransformer

OP_DIRS = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x_mul_y",
    "div": "x_div_y",
}
DEFAULT_SEEDS = [0, 1, 2]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_tensors(p, operation, run_dir, project_root, split):
    legacy_path = os.path.join(project_root, "data", OP_DIRS[operation], f"{split}_data.json")
    seed_path = os.path.join(run_dir, f"{split}_data.json")
    data_path = seed_path if os.path.exists(seed_path) else legacy_path
    with open(data_path, "r") as f:
        pairs = json.load(f)

    labels = []
    for x, y in pairs:
        if operation == "add":
            labels.append((x + y) % p)
        elif operation == "sub":
            labels.append((x - y) % p)
        elif operation == "mul":
            labels.append((x * y) % p)
        elif operation == "div":
            labels.append(0 if y == 0 else (x * pow(y, -1, p)) % p)

    inputs = torch.tensor([[x, p, y] for x, y in pairs], dtype=torch.long)
    return inputs, torch.tensor(labels, dtype=torch.long)


def load_theta(project_root, op_dir, run_dir, step, device):
    path = os.path.join(run_dir, "checkpoints", f"checkpoint_step_{step}.pt")
    state = torch.load(path, map_location="cpu")
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    return {k: v.to(device).float() for k, v in sd.items()}


def curve_point(theta_a, theta_b, bend, t):
    ca, cm, cb = (1 - t) ** 2, 2 * t * (1 - t), t ** 2
    return {k: ca * theta_a[k] + cm * bend[k] + cb * theta_b[k] for k in theta_a}


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


def train_bend(model, theta_a, theta_b, inputs, labels, device, args, seed):
    """Train the Bezier bend point on plain train cross-entropy."""
    bend = {k: ((theta_a[k] + theta_b[k]) / 2).clone().requires_grad_(True)
            for k in theta_a}
    opt = torch.optim.Adam(bend.values(), lr=args.curve_lr)
    generator = torch.Generator(device="cpu").manual_seed(seed * 7919 + 13)
    n = inputs.shape[0]

    model.eval()
    for step in range(args.curve_steps):
        idx = torch.randint(0, n, (args.batch_size,), generator=generator)
        xb, yb = inputs[idx].to(device), labels[idx].to(device)
        t = float(torch.rand(1, generator=generator))
        params = curve_point(theta_a, theta_b, bend, t)
        logits = functional_call(model, params, (xb,))
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(1, args.curve_steps // 5) == 0:
            print(f"    curve step {step + 1}/{args.curve_steps}  t={t:.3f}  loss={loss.item():.6f}")
    return {k: v.detach() for k, v in bend.items()}


@torch.no_grad()
def eval_params(model, params, inputs, labels, device, batch_size=4096):
    total_loss, correct, margin_sum, n = 0.0, 0, 0.0, inputs.shape[0]
    for start in range(0, n, batch_size):
        xb = inputs[start:start + batch_size].to(device)
        yb = labels[start:start + batch_size].to(device)
        logits = functional_call(model, params, (xb,))
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        correct_logit = logits.gather(1, yb.unsqueeze(1)).squeeze(1)
        tmp = logits.clone()
        tmp.scatter_(1, yb.unsqueeze(1), float("-inf"))
        margin_sum += (correct_logit - tmp.max(dim=1).values).sum().item()
    return total_loss / n, correct / n, margin_sum / n


def l2_norm(params):
    return float(torch.sqrt(sum(p.pow(2).sum() for p in params.values())).item())


def evaluate_path(model, theta_a, theta_b, bend, path_name, ts, data, device, wd):
    (train_x, train_y), (test_x, test_y) = data
    rows = []
    for t in ts:
        if path_name == "linear":
            params = {k: (1 - t) * theta_a[k] + t * theta_b[k] for k in theta_a}
        else:
            params = curve_point(theta_a, theta_b, bend, t)
        train_loss, train_acc, train_margin = eval_params(model, params, train_x, train_y, device)
        test_loss, test_acc, test_margin = eval_params(model, params, test_x, test_y, device)
        norm = l2_norm(params)
        rows.append({
            "path": path_name, "t": t,
            "train_loss": train_loss, "train_acc": train_acc, "train_margin": train_margin,
            "test_loss": test_loss, "test_acc": test_acc, "test_margin": test_margin,
            "l2_norm": norm,
            "reg_train_loss": train_loss + 0.5 * wd * norm * norm,
        })
    return rows


def barrier(rows, col):
    """max over t of value minus the endpoint chord."""
    ts = np.array([r["t"] for r in rows])
    vals = np.array([r[col] for r in rows])
    chord = (1 - ts) * vals[0] + ts * vals[-1]
    return float(np.max(vals - chord))


def process_seed(args, config, project_root, op_dir, seed, device, aggregate_writer=None):
    set_seed(seed)
    run_dir = resolve_seed_run_dir(project_root, op_dir, seed, args.multi_seed)
    for step in (args.start, args.end):
        if not os.path.exists(os.path.join(run_dir, "checkpoints", f"checkpoint_step_{step}.pt")):
            print(f"[SKIP] seed={seed}: checkpoint_step_{step}.pt not found in {run_dir}")
            return

    print(f"seed={seed}  run_dir={run_dir}  endpoints={args.start}->{args.end}")
    theta_a = load_theta(project_root, op_dir, run_dir, args.start, device)
    theta_b = load_theta(project_root, op_dir, run_dir, args.end, device)

    train_x, train_y = load_split_tensors(config.p, args.operation, run_dir, project_root, "train")
    test_x, test_y = load_split_tensors(config.p, args.operation, run_dir, project_root, "test")
    data = ((train_x, train_y), (test_x, test_y))

    model = GrokkingTransformer(config).to(device)
    model.eval()

    print("  training Bezier bend point ...")
    bend = train_bend(model, theta_a, theta_b, train_x, train_y, device, args, seed)

    ts = np.linspace(0.0, 1.0, args.num_points)
    rows = []
    for path_name in ("linear", "bezier"):
        rows.extend(evaluate_path(model, theta_a, theta_b, bend, path_name, ts, data, device, args.wd))

    out_csv = os.path.join(run_dir, "mode_connectivity.csv")
    fields = ["path", "t", "train_loss", "train_acc", "train_margin",
              "test_loss", "test_acc", "test_margin", "l2_norm", "reg_train_loss"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  saved: {out_csv}")

    if aggregate_writer is not None:
        for r in rows:
            out = {k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in r.items()}
            out["seed"] = seed
            aggregate_writer.writerow(out)

    for path_name in ("linear", "bezier"):
        sub = [r for r in rows if r["path"] == path_name]
        print(f"  [{path_name:6s}] train-loss barrier = {barrier(sub, 'train_loss'):.6f} | "
              f"reg-train-loss barrier = {barrier(sub, 'reg_train_loss'):.6f} | "
              f"min test_acc on path = {min(r['test_acc'] for r in sub):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Bezier mode connectivity between grokking phases")
    parser.add_argument("--operation", default="add", choices=list(OP_DIRS))
    parser.add_argument("--start", type=int, default=2000, help="Memorization checkpoint step.")
    parser.add_argument("--end", type=int, default=99000, help="Generalization checkpoint step.")
    parser.add_argument("--curve-steps", type=int, default=3000)
    parser.add_argument("--curve-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-points", type=int, default=201)
    parser.add_argument("--wd", type=float, default=Config.weight_decay,
                        help="Weight decay used for the regularized potential.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    config = Config(args.operation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIRS[args.operation]
    seeds = args.seeds if args.multi_seed else [args.seed]

    aggregate_file, aggregate_writer = None, None
    if args.multi_seed:
        aggregate_path = os.path.join(project_root, "data", op_dir, "mode_connectivity_multiseed.csv")
        aggregate_file = open(aggregate_path, "w", newline="")
        fields = ["path", "t", "train_loss", "train_acc", "train_margin",
                  "test_loss", "test_acc", "test_margin", "l2_norm", "reg_train_loss", "seed"]
        aggregate_writer = csv.DictWriter(aggregate_file, fieldnames=fields)
        aggregate_writer.writeheader()

    try:
        for seed in seeds:
            process_seed(args, config, project_root, op_dir, seed, device, aggregate_writer)
    finally:
        if aggregate_file is not None:
            aggregate_file.close()
            print(f"\nMulti-seed data saved to: data/{op_dir}/mode_connectivity_multiseed.csv")


if __name__ == "__main__":
    main()
