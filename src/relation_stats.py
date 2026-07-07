#!/usr/bin/env python3
"""
Pooled head-level structure–CMA correlation statistics.

Addresses the n=8 concern by pooling heads across seeds and configs,
with permutation tests and bootstrap confidence intervals.

Reads:
    data/x+y/robustness/{cfg}/seed_{s}/relation/relation_heads.csv
    data/x+y/seed_{s}/relation/relation_heads.csv  (baseline)

Writes:
    data/x+y/robustness/relation_stats.csv
    data/x+y/robustness/relation_stats_pooled.csv

Usage:
    python src/relation_stats.py
    python src/relation_stats.py --steps 90000
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relation import rankdata, safe_corr
from robustness_common import (
    DEFAULT_SEEDS,
    OP_DIR,
    SIGNATURE_STEPS,
    discover_robustness_runs,
    robustness_root,
)


def load_head_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def bootstrap_ci(x, y, n_boot=5000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(safe_corr(x[idx], y[idx]))
    stats = np.asarray([s for s in stats if not np.isnan(s)])
    if len(stats) == 0:
        return np.nan, np.nan, np.nan
    alpha = (1 - ci) / 2
    return float(np.mean(stats)), float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def permutation_pvalue(x, y, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = safe_corr(x, y)
    if np.isnan(obs):
        return np.nan, obs
    count = 0
    for _ in range(n_perm):
        perm_y = rng.permutation(y)
        if abs(safe_corr(x, perm_y)) >= abs(obs):
            count += 1
    return count / n_perm, obs


def collect_runs(project_root, operation, baseline_seeds):
    op_dir = OP_DIR[operation]
    runs = []

    # baseline (existing multi-seed)
    for seed in baseline_seeds:
        path = os.path.join(project_root, "data", op_dir, f"seed_{seed}", "relation", "relation_heads.csv")
        if os.path.exists(path):
            runs.append(("baseline", seed, path))

    # robustness sweep
    for config_name, seed, run_dir in discover_robustness_runs(project_root, operation):
        path = os.path.join(run_dir, "relation", "relation_heads.csv")
        if os.path.exists(path):
            runs.append((config_name, seed, path))
    return runs


def filter_step(rows, step):
    return [r for r in rows if int(r["step"]) == step]


def analyze_pool(label, rows, step, n_boot, n_perm):
    subset = filter_step(rows, step)
    if len(subset) < 3:
        return None

    structure = [float(r["structure_score"]) for r in subset]
    cma = [float(r["cma_score"]) for r in subset]
    n_heads = len(subset)

    pearson = safe_corr(structure, cma)
    spearman = safe_corr(rankdata(structure), rankdata(cma))
    p_mean, p_lo, p_hi = bootstrap_ci(structure, cma, n_boot=n_boot)
    s_mean, s_lo, s_hi = bootstrap_ci(rankdata(structure), rankdata(cma), n_boot=n_boot, seed=1)
    p_perm, _ = permutation_pvalue(structure, cma, n_perm=n_perm)
    s_perm, _ = permutation_pvalue(rankdata(structure), rankdata(cma), n_perm=n_perm, seed=2)

    return {
        "pool": label,
        "step": step,
        "n_heads": n_heads,
        "pearson": pearson,
        "spearman": spearman,
        "pearson_boot_mean": p_mean,
        "pearson_ci_lo": p_lo,
        "pearson_ci_hi": p_hi,
        "spearman_boot_mean": s_mean,
        "spearman_ci_lo": s_lo,
        "spearman_ci_hi": s_hi,
        "pearson_perm_p": p_perm,
        "spearman_perm_p": s_perm,
    }


def per_run_stats(config_name, seed, path, steps):
    rows = load_head_rows(path)
    out = []
    for step in steps:
        subset = filter_step(rows, step)
        if len(subset) < 2:
            continue
        structure = [float(r["structure_score"]) for r in subset]
        cma = [float(r["cma_score"]) for r in subset]
        out.append({
            "config": config_name,
            "seed": seed,
            "step": step,
            "n_heads": len(subset),
            "pearson": safe_corr(structure, cma),
            "spearman": safe_corr(rankdata(structure), rankdata(cma)),
        })
    return out


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted[k] = "" if np.isnan(v) else f"{v:.6f}"
                else:
                    formatted[k] = v
            writer.writerow(formatted)
    print(f"[OK] {path}")


def main():
    parser = argparse.ArgumentParser(description="Pooled head-level relation statistics")
    parser.add_argument("--operation", default="add", choices=list(OP_DIR.keys()))
    parser.add_argument("--steps", type=int, nargs="+", default=SIGNATURE_STEPS)
    parser.add_argument("--baseline-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    op_dir = OP_DIR[args.operation]
    out_dir = robustness_root(project_root, args.operation)

    runs = collect_runs(project_root, args.operation, args.baseline_seeds)

    per_run_rows = []
    for config_name, seed, path in runs:
        per_run_rows.extend(per_run_stats(config_name, seed, path, args.steps))

    all_rows = []
    for _, _, path in runs:
        all_rows.extend(load_head_rows(path))

    baseline_rows = []
    for config_name, seed, path in runs:
        if config_name == "baseline":
            baseline_rows.extend(load_head_rows(path))

    robust_rows = []
    for config_name, seed, path in runs:
        if config_name != "baseline":
            robust_rows.extend(load_head_rows(path))

    pooled_rows = []
    for step in args.steps:
        for label, pool in [
            ("baseline_only", baseline_rows),
            ("robustness_only", robust_rows),
            ("all_pooled", all_rows),
        ]:
            result = analyze_pool(label, pool, step, args.n_bootstrap, args.n_perm)
            if result is not None:
                pooled_rows.append(result)
                print(
                    f"step={step} pool={label} n={result['n_heads']} "
                    f"pearson={result['pearson']:.3f} (p={result['pearson_perm_p']:.4f}) "
                    f"CI=[{result['pearson_ci_lo']:.3f}, {result['pearson_ci_hi']:.3f}]"
                )

    if per_run_rows:
        write_csv(
            os.path.join(out_dir, "relation_stats.csv"),
            per_run_rows,
            list(per_run_rows[0].keys()),
        )
    if pooled_rows:
        write_csv(
            os.path.join(out_dir, "relation_stats_pooled.csv"),
            pooled_rows,
            list(pooled_rows[0].keys()),
        )


if __name__ == "__main__":
    main()
