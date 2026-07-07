#!/usr/bin/env python3
"""Shared helpers for hyperparameter robustness sweep."""

import json
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, OP_DIR

# OFAT configs around the add baseline (wd=0.005, init=1, split=0.5, 2L/128d).
ROBUSTNESS_CONFIGS = [
    {"name": "wd_0.001", "overrides": {"weight_decay": 0.001}},
    {"name": "wd_0.02", "overrides": {"weight_decay": 0.02}},
    {"name": "wd_0.1", "overrides": {"weight_decay": 0.1}},
    {"name": "init_0.5", "overrides": {"init_scale": 0.5}},
    {"name": "init_2.0", "overrides": {"init_scale": 2.0}},
    {"name": "split_0.3", "overrides": {"train_ratio": 0.3}},
    {"name": "split_0.7", "overrides": {"train_ratio": 0.7}},
    {"name": "layers_1", "overrides": {"num_layers": 1}},
    {"name": "layers_4", "overrides": {"num_layers": 4}},
    {"name": "width_64", "overrides": {"attention_dim": 64, "ffn_dim": 256}},
    {"name": "width_256", "overrides": {"attention_dim": 256, "ffn_dim": 1024}},
]

DEFAULT_SEEDS = [0, 1]
SIGNATURE_STEPS = [1000, 5000, 10000, 30000, 90000]

CONFIG_KEYS = [
    "weight_decay", "init_scale", "train_ratio", "num_layers",
    "attention_dim", "ffn_dim", "num_heads", "total_steps",
]


def robustness_root(project_root, operation="add"):
    op_dir = OP_DIR.get(operation, operation)
    return os.path.join(project_root, "data", op_dir, "robustness")


def run_dir_for(project_root, operation, config_name, seed):
    return os.path.join(robustness_root(project_root, operation), config_name, f"seed_{seed}")


def load_config_from_run_dir(run_dir, operation="add"):
    """Build a Config matching the saved training run (architecture + hyperparams)."""
    overrides = {}
    seed = 0
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            snapshot = json.load(f)
        seed = snapshot.get("seed", 0)
        for key in CONFIG_KEYS:
            if key in snapshot:
                overrides[key] = snapshot[key]
    return Config(operation, seed=seed, run_dir=run_dir, overrides=overrides)


def discover_robustness_runs(project_root, operation="add", seeds=None):
    """Return list of (config_name, seed, run_dir) for completed/in-progress runs."""
    root = robustness_root(project_root, operation)
    runs = []
    if not os.path.isdir(root):
        return runs

    for entry in sorted(os.listdir(root)):
        if entry.startswith("_"):
            continue
        cfg_dir = os.path.join(root, entry)
        if not os.path.isdir(cfg_dir):
            continue
        for seed_entry in sorted(os.listdir(cfg_dir)):
            if not seed_entry.startswith("seed_"):
                continue
            seed = int(seed_entry.split("_")[1])
            if seeds is not None and seed not in seeds:
                continue
            run_dir = os.path.join(cfg_dir, seed_entry)
            if os.path.isdir(os.path.join(run_dir, "checkpoints")):
                runs.append((entry, seed, run_dir))
    return runs


def load_metric_csv(path):
    import csv

    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append({
                "step": int(row["step"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "test_loss": float(row["test_loss"]),
                "test_acc": float(row["test_acc"]),
            })
    return rows


def detect_grokking_step(metric_rows, threshold=0.9, min_step=500):
    """First step where test accuracy crosses threshold (after min_step)."""
    for row in metric_rows:
        if row["step"] >= min_step and row["test_acc"] >= threshold:
            return row["step"]
    return None


def final_test_acc(metric_rows):
    if not metric_rows:
        return None
    return metric_rows[-1]["test_acc"]
