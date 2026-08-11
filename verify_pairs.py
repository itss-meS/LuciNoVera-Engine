import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.npy_dataset import load_npy_as_hw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--out", default="results/comparison_samples/pair_check.png")
    return p.parse_args()


def main():
    args = parse_args()
    gt_files = sorted(f for f in os.listdir(args.gt_dir) if f.endswith(".npy"))[:args.n]
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(len(gt_files), 3, figsize=(9, 3 * len(gt_files)))
    if len(gt_files) == 1:
        axes = axes[None, :]

    for row, name in enumerate(gt_files):
        gt_path = os.path.join(args.gt_dir, name)
        deg_path = os.path.join(args.degraded_dir, name)

        gt = load_npy_as_hw(gt_path, mmap=False)
        deg = load_npy_as_hw(deg_path, mmap=False) if os.path.exists(deg_path) else None

        print(f"{name}: GT shape={gt.shape} dtype={gt.dtype} range=[{gt.min():.3f},{gt.max():.3f}]")
        if deg is not None:
            print(f"{name}: DEG shape={deg.shape} dtype={deg.dtype} range=[{deg.min():.3f},{deg.max():.3f}]")
        else:
            print(f"{name}: NO MATCHING DEGRADED FILE FOUND at {deg_path}")

        axes[row, 0].imshow(gt, cmap="gray")
        axes[row, 0].set_title(f"GT: {name}")
        if deg is not None:
            axes[row, 1].imshow(deg, cmap="gray")
            axes[row, 1].set_title("Degraded")
            axes[row, 2].hist(gt.ravel(), bins=50, alpha=0.5, label="gt")
            axes[row, 2].hist(deg.ravel(), bins=50, alpha=0.5, label="degraded")
            axes[row, 2].legend()
        for ax in axes[row, :2]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved comparison figure to {args.out}")


if __name__ == "__main__":
    main()
