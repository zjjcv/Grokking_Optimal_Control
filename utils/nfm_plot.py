#!/usr/bin/env python3
"""
Plot Neural Feature Map (NFM) heatmaps.

Reads data/{op}/nfm/nfm_{component}_step_{N}.npy and produces heatmaps
in results/{op}/nfm/.

Usage:
    python utils/nfm_plot.py --operation add
    python utils/nfm_plot.py --operation all
    python utils/nfm_plot.py --operation mul --view embedding unembedding
"""

import os
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


OPS = {
    "add": ("x+y", "+"),
    "sub": ("x-y", "-"),
    "mul": ("x_mul_y", r"\times"),
    "div": ("x_div_y", r"\div"),
}


def primitive_root(p):
    """Find a primitive root of prime p."""
    if p == 2:
        return 1
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


def dlog_permutation(p):
    """Compute discrete log reordering for multiplicative group mod p.

    Returns perm where perm[k] = g^k mod p for k=0..p-2, perm[p-1]=0.
    Applying M[np.ix_(perm, perm)] makes multiplicative structure diagonal.
    """
    g = primitive_root(p)
    perm = np.zeros(p, dtype=int)
    val = 1
    for k in range(p - 1):
        perm[k] = val
        val = (val * g) % p
    perm[p - 1] = 0
    return perm


# Component display labels
COMPONENT_LABELS = {
    "embedding": r"$M_E = EE^\top$",
    "unembedding": r"$M_U = UU^\top$",
    "mlp_in_l0": r"$M_{\mathrm{MLP,in}}^{(0)}$",
    "mlp_in_l1": r"$M_{\mathrm{MLP,in}}^{(1)}$",
    "mlp_out_l0": r"$M_{\mathrm{MLP,out}}^{(0)}$",
    "mlp_out_l1": r"$M_{\mathrm{MLP,out}}^{(1)}$",
    "qk_l0_h0": r"$M_{QK}^{(0,h0)}$",
    "qk_l0_h1": r"$M_{QK}^{(0,h1)}$",
    "qk_l0_h2": r"$M_{QK}^{(0,h2)}$",
    "qk_l0_h3": r"$M_{QK}^{(0,h3)}$",
    "qk_l1_h0": r"$M_{QK}^{(1,h0)}$",
    "qk_l1_h1": r"$M_{QK}^{(1,h1)}$",
    "qk_l1_h2": r"$M_{QK}^{(1,h2)}$",
    "qk_l1_h3": r"$M_{QK}^{(1,h3)}$",
    "ov_l0_h0": r"$M_{OV}^{(0,h0)}$",
    "ov_l0_h1": r"$M_{OV}^{(0,h1)}$",
    "ov_l0_h2": r"$M_{OV}^{(0,h2)}$",
    "ov_l0_h3": r"$M_{OV}^{(0,h3)}$",
    "ov_l1_h0": r"$M_{OV}^{(1,h0)}$",
    "ov_l1_h1": r"$M_{OV}^{(1,h1)}$",
    "ov_l1_h2": r"$M_{OV}^{(1,h2)}$",
    "ov_l1_h3": r"$M_{OV}^{(1,h3)}$",
}

# Component grouping
COMPONENT_GROUPS = {
    "embedding": ["embedding"],
    "unembedding": ["unembedding"],
    "mlp_in": ["mlp_in_l0", "mlp_in_l1"],
    "mlp_out": ["mlp_out_l0", "mlp_out_l1"],
    "qk": [f"qk_l{l}_h{h}" for l in range(2) for h in range(4)],
    "ov": [f"ov_l{l}_h{h}" for l in range(2) for h in range(4)],
}

ALL_COMPONENTS = ["embedding", "unembedding",
                  "mlp_in_l0", "mlp_in_l1", "mlp_out_l0", "mlp_out_l1",
                  "qk_l0_h0", "qk_l0_h1", "qk_l0_h2", "qk_l0_h3",
                  "qk_l1_h0", "qk_l1_h1", "qk_l1_h2", "qk_l1_h3",
                  "ov_l0_h0", "ov_l0_h1", "ov_l0_h2", "ov_l0_h3",
                  "ov_l1_h0", "ov_l1_h1", "ov_l1_h2", "ov_l1_h3"]


def robust_symmetric_norm(A, q=99.5):
    B = A - np.median(A)
    vmax = np.percentile(np.abs(B), q)
    if vmax <= 1e-12:
        vmax = np.max(np.abs(B)) + 1e-12
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    return B, norm


def circulant_alignment(A):
    """Frobenius cosine between A and its projection onto circulant matrices."""
    M = A - np.mean(A)
    denom = np.linalg.norm(M, ord="fro")
    if denom <= 1e-12:
        return 0.0

    n = M.shape[0]
    C = np.zeros_like(M)
    for offset in range(n):
        vals = M[np.arange(n), (np.arange(n) + offset) % n]
        C[np.arange(n), (np.arange(n) + offset) % n] = np.mean(vals)

    return float(np.linalg.norm(C, ord="fro") / denom)


def component_title(component, mat):
    label = COMPONENT_LABELS.get(component, component)
    if component.startswith("ov_"):
        return f"{label}\nCirc. align={circulant_alignment(mat):.3f}"
    return label


def plot_nfm(component, step, mat, p, out_path, perm=None):
    """Plot a single NFM heatmap."""
    M = mat
    if perm is not None:
        M = mat[np.ix_(perm, perm)]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    B, norm = robust_symmetric_norm(M)
    im = ax.imshow(B, cmap="RdBu_r", norm=norm, aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if perm is not None:
        ax.set_xticks([0, p - 1])
        ax.set_xticklabels([str(perm[0]), str(perm[p - 1])])
        ax.set_yticks([0, p - 1])
        ax.set_yticklabels([str(perm[0]), str(perm[p - 1])])
        ax.set_xlabel("Token (dlog order)")
        ax.set_ylabel("Token (dlog order)")
    else:
        ax.set_xticks([0, p - 1])
        ax.set_xticklabels(["0", f"{p - 1}"])
        ax.set_yticks([0, p - 1])
        ax.set_yticklabels(["0", f"{p - 1}"])
        ax.set_xlabel("Token index")
        ax.set_ylabel("Token index")

    label = component_title(component, M)
    tag = " (dlog)" if perm is not None else ""
    ax.set_title(f"{label}{tag} — Step {step}", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()


def plot_group_grid(group_name, components, step, data_dict, p, out_dir, perm=None):
    """Plot a group of components in a single grid figure."""
    n = len(components)
    if n == 1:
        comp = components[0]
        if comp not in data_dict:
            return
        path = os.path.join(out_dir, f"nfm_{comp}_step_{step}.svg")
        plot_nfm(comp, step, data_dict[comp], p, path, perm=perm)
        print(f"[OK] {path}")
        return

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, comp in enumerate(components):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if comp not in data_dict:
            ax.set_visible(False)
            continue
        mat = data_dict[comp]
        if perm is not None:
            mat = mat[np.ix_(perm, perm)]
        B, norm = robust_symmetric_norm(mat)
        im = ax.imshow(B, cmap="RdBu_r", norm=norm, aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        label = component_title(comp, mat)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([0, p - 1])
        ax.set_yticks([0, p - 1])

    for idx in range(len(components), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Step {step}", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"nfm_{group_name}_step_{step}.svg")
    plt.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[OK] {path}")


def process_operation(op_key, project_root, view_groups, p):
    op_dir, symbol = OPS[op_key]
    nfm_dir = os.path.join(project_root, "data", op_dir, "nfm")
    out_dir = os.path.join(project_root, "results", op_dir, "nfm")

    if not os.path.isdir(nfm_dir):
        print(f"[SKIP] {nfm_dir} not found")
        return

    os.makedirs(out_dir, exist_ok=True)

    perm = dlog_permutation(p) if op_key in ("mul", "div") else None

    # Discover available steps
    npy_files = [f for f in os.listdir(nfm_dir) if f.endswith(".npy")]
    step_re = re.compile(r"nfm_.+_step_(\d+)\.npy")
    steps = sorted(set(int(step_re.search(f).group(1)) for f in npy_files if step_re.search(f)))

    if not steps:
        print(f"[SKIP] No NFM .npy files in {nfm_dir}")
        return

    # Determine which groups to plot
    if view_groups is None:
        view_groups = list(COMPONENT_GROUPS.keys())

    for step in steps:
        # Load all components for this step
        data = {}
        for comp in ALL_COMPONENTS:
            path = os.path.join(nfm_dir, f"nfm_{comp}_step_{step}.npy")
            if os.path.exists(path):
                data[comp] = np.load(path)

        print(f"\nStep {step} | {len(data)} components loaded")

        for group in view_groups:
            if group not in COMPONENT_GROUPS:
                print(f"  [WARN] Unknown group: {group}")
                continue
            components = COMPONENT_GROUPS[group]
            available = [c for c in components if c in data]
            if not available:
                continue

            if len(available) == 1:
                comp = available[0]
                path = os.path.join(out_dir, f"nfm_{comp}_step_{step}.svg")
                plot_nfm(comp, step, data[comp], p, path, perm=perm)
                print(f"[OK] {path}")
            else:
                plot_group_grid(group, available, step, data, p, out_dir, perm=perm)


def main():
    parser = argparse.ArgumentParser(description="Plot NFM heatmaps")
    parser.add_argument("--operation", choices=list(OPS.keys()) + ["all"], default="all")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--view", nargs="*", default=None,
                        help="Component groups to plot: embedding unembedding mlp_in mlp_out qk ov. Default: all")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ops = list(OPS.keys()) if args.operation == "all" else [args.operation]
    for op in ops:
        process_operation(op, project_root, args.view, args.p)


if __name__ == "__main__":
    main()
