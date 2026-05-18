#!/usr/bin/env python3
"""
绘制 paper-style AGOP 矩阵热力图。

重点：
    1. full:
        画完整 2p x 2p AGOP。
        这会被 G_xx/G_yy 主导，cross block 可能显白。

    2. cross:
        单独画右上 cross block G_xy。
        默认做 y -> -y mod p，使 x+y=const 变成正对角线族。
        使用 cross block 自己的 robust 色阶。

    3. cross_full:
        画只含 cross block 的 2p x 2p 矩阵：
            [[0, G_xy_neg_y],
             [G_xy_neg_y.T, 0]]
        这样左上、右下置零，右上、左下结构会显现。

数据来源:
    data/{op}/agop_paper/agop_paper_step_{N}_raw.csv
    或 .npy

输出:
    results/{op}/agop/*.pdf

用法:
    python src/plot_agop.py --operation add --view all
    python src/plot_agop.py --operation add --view cross
    python src/plot_agop.py --operation add --view cross_full
    python src/plot_agop.py --operation add --view cross --project-circulant
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


OPS = {
    "add": "x+y",
    "sub": "x-y",
    "mul": "x*y",
    "div": "x_div_y",
}


def read_matrix(path):
    if path.endswith(".npy"):
        return np.load(path)

    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(v) for v in row])
    return np.array(data)


def robust_symmetric_norm(A, q=99.5, center=True):
    """
    对矩阵 A 使用 robust symmetric color scale。

    center=True:
        先减去 median，避免整体偏置导致正负结构不明显。
    """
    B = A.copy()

    if center:
        B = B - np.median(B)

    vmax = np.percentile(np.abs(B), q)

    if vmax <= 1e-12:
        vmax = np.max(np.abs(B)) + 1e-12

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    return B, norm


def negate_y_axis(A):
    """
    对 cross block A[x, y] 的 y 轴做 y -> -y mod p。

    原始结构:
        x + y = const

    令 y' = -y，则：
        x - y' = const

    视觉上反对角线族变成正对角线族。
    """
    p = A.shape[0]
    perm = (-np.arange(p)) % p
    return A[:, perm]


def circulant_projection(A):
    """
    将矩阵投影到 wrapped diagonal / circulant 子空间。

    C[i, (i+d) mod p] = mean_i A[i, (i+d) mod p]

    这不是原始矩阵，而是提取其正对角线族成分。
    """
    p = A.shape[0]
    C = np.zeros_like(A)

    for d in range(p):
        vals = np.array([A[i, (i + d) % p] for i in range(p)])
        mean_val = vals.mean()
        for i in range(p):
            C[i, (i + d) % p] = mean_val

    return C


def diagonal_snr(A):
    """
    wrapped diagonal signal-to-noise ratio。

    数值越大，说明矩阵越接近 wrapped diagonal / circulant 条纹。
    """
    p = A.shape[0]

    diag_means = []
    diag_vars = []

    for d in range(p):
        vals = np.array([A[i, (i + d) % p] for i in range(p)])
        diag_means.append(vals.mean())
        diag_vars.append(vals.var())

    return float(np.var(diag_means) / (np.mean(diag_vars) + 1e-12))


def add_block_guides(ax, p):
    ax.axhline(y=p - 0.5, color="black", linewidth=1.0)
    ax.axvline(x=p - 0.5, color="black", linewidth=1.0)

    ax.text(p / 2, -8, "x", ha="center", fontsize=11, fontweight="bold")
    ax.text(p + p / 2, -8, "y", ha="center", fontsize=11, fontweight="bold")
    ax.text(
        -8,
        p / 2,
        "x",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        rotation=90,
    )
    ax.text(
        -8,
        p + p / 2,
        "y",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        rotation=90,
    )

    ticks = [0, p - 1, p, 2 * p - 1]
    tick_labels = ["0", f"{p - 1}", "0", f"{p - 1}"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)


def plot_full(step, agop, out_path, p=97, q=99.9):
    """
    原始完整 AGOP 图。
    使用全矩阵色阶，因此 cross block 可能仍较弱。
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    A, norm = robust_symmetric_norm(agop, q=q, center=True)

    im = ax.imshow(
        A,
        cmap="RdBu_r",
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    add_block_guides(ax, p)

    ax.set_xlabel("Input dimension (one-hot)")
    ax.set_ylabel("Input dimension (one-hot)")
    ax.set_title(f"AGOP Matrix — Step {step}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] {out_path}")


def plot_cross(step, agop, out_path, p=97, q=99.5, negate_y=True, project_circulant=False):
    """
    单独画右上 cross block G_xy。

    这是最推荐的图。
    """
    G_xy = agop[:p, p:]

    print(f"[Step {step}] raw G_xy range = [{G_xy.min():.4e}, {G_xy.max():.4e}]")
    print(f"[Step {step}] raw G_xy fro   = {np.linalg.norm(G_xy):.4e}")
    print(f"[Step {step}] raw diag SNR   = {diagonal_snr(G_xy):.4e}")

    A = G_xy.copy()

    if negate_y:
        A = negate_y_axis(A)
        print(f"[Step {step}] after y->-y diag SNR = {diagonal_snr(A):.4e}")

    if project_circulant:
        A = circulant_projection(A)
        print(f"[Step {step}] projected diag SNR  = {diagonal_snr(A):.4e}")

    A_plot, norm = robust_symmetric_norm(A, q=q, center=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(
        A_plot,
        cmap="RdBu_r",
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, p - 1])
    ax.set_xticklabels(["0", f"{p - 1}"])
    ax.set_yticks([0, p - 1])
    ax.set_yticklabels(["0", f"{p - 1}"])

    ax.set_xlabel("-y coordinate" if negate_y else "y coordinate")
    ax.set_ylabel("x coordinate")

    title = f"AGOP Cross Block $G_{{xy}}$ — Step {step}"
    if negate_y:
        title += " | y→−y"
    if project_circulant:
        title += " | projected"
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[OK] {out_path}")


def plot_cross_full(step, agop, out_path, p=97, q=99.5, negate_y=True, project_circulant=False):
    """
    画只含 cross block 的 2p x 2p 矩阵：

        [[0, A],
         [A.T, 0]]

    其中 A = G_xy 或 G_xy after y->-y。
    """
    G_xy = agop[:p, p:]
    A = G_xy.copy()

    if negate_y:
        A = negate_y_axis(A)

    if project_circulant:
        A = circulant_projection(A)

    M = np.zeros_like(agop)
    M[:p, p:] = A
    M[p:, :p] = A.T

    M_plot, norm = robust_symmetric_norm(M, q=q, center=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        M_plot,
        cmap="RdBu_r",
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    add_block_guides(ax, p)

    ax.set_xlabel("Input dimension (one-hot)")
    ax.set_ylabel("Input dimension (one-hot)")

    title = f"AGOP Cross-only Matrix — Step {step}"
    if negate_y:
        title += " | y→−y"
    if project_circulant:
        title += " | projected"
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"[OK] {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--operation", default="add", choices=["add", "sub", "mul", "div"])

    parser.add_argument(
        "--view",
        default="all",
        choices=["full", "cross", "cross_full", "all"],
        help="full: full AGOP; cross: G_xy only; cross_full: zero diagonal blocks; all: generate all.",
    )

    parser.add_argument(
        "--kind",
        default="raw",
        choices=["raw", "sqrt"],
        help="Use raw or sqrt AGOP file.",
    )

    parser.add_argument("--p", type=int, default=97)

    parser.add_argument(
        "--q",
        type=float,
        default=99.5,
        help="Robust percentile for color scale.",
    )

    parser.add_argument(
        "--no-negate-y",
        action="store_true",
        help="Do not apply y -> -y to cross block.",
    )

    parser.add_argument(
        "--project-circulant",
        action="store_true",
        help="Project cross block to wrapped diagonal/circulant component.",
    )

    parser.add_argument(
        "--agop-subdir",
        default="agop_paper",
        help="Subdirectory under data/{op}/ containing AGOP files.",
    )

    args = parser.parse_args()

    op_dir = OPS[args.operation]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    agop_dir = os.path.join(project_root, "data", op_dir, args.agop_subdir)
    out_dir = os.path.join(project_root, "results", op_dir, "agop")
    os.makedirs(out_dir, exist_ok=True)

    p = args.p
    negate_y = not args.no_negate_y

    suffix = f"_{args.kind}.csv"

    csv_files = sorted(
        [
            f
            for f in os.listdir(agop_dir)
            if f.startswith("agop_paper_step_") and f.endswith(suffix)
        ],
        key=lambda x: int(x.split("_step_")[1].split("_")[0]),
    )

    if not csv_files:
        print(f"No AGOP CSV files found in {agop_dir} with suffix {suffix}")
        return

    for csv_file in csv_files:
        step = int(csv_file.split("_step_")[1].split("_")[0])
        agop_path = os.path.join(agop_dir, csv_file)
        agop = read_matrix(agop_path)

        tag = args.kind
        if negate_y:
            tag += "_negy"
        else:
            tag += "_nonegy"
        if args.project_circulant:
            tag += "_circproj"

        if args.view in ["full", "all"]:
            pdf_path = os.path.join(out_dir, f"agop_full_{tag}_step_{step}.pdf")
            plot_full(step, agop, pdf_path, p=p, q=max(args.q, 99.9))

        if args.view in ["cross", "all"]:
            pdf_path = os.path.join(out_dir, f"agop_cross_{tag}_step_{step}.pdf")
            plot_cross(
                step,
                agop,
                pdf_path,
                p=p,
                q=args.q,
                negate_y=negate_y,
                project_circulant=args.project_circulant,
            )

        if args.view in ["cross_full", "all"]:
            pdf_path = os.path.join(out_dir, f"agop_cross_full_{tag}_step_{step}.pdf")
            plot_cross_full(
                step,
                agop,
                pdf_path,
                p=p,
                q=args.q,
                negate_y=negate_y,
                project_circulant=args.project_circulant,
            )


if __name__ == "__main__":
    main()