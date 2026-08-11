import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--degraded_dir", default="dataset/test/degraded")
    p.add_argument("--restored_dir", default="results/test_restored")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    restored_files = sorted(f for f in os.listdir(args.restored_dir) if f.endswith(".npy"))
    if not restored_files:
        raise RuntimeError(f"No .npy files found in {args.restored_dir}")

    n = min(args.n, len(restored_files))
    chosen = sorted(random.sample(restored_files, n))

    state = {"idx": 0}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.15)

    def show_current():
        name = chosen[state["idx"]]
        degraded = np.load(os.path.join(args.degraded_dir, name))
        restored = np.load(os.path.join(args.restored_dir, name))

        axes[0].clear()
        axes[1].clear()

        axes[0].imshow(degraded, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Degraded: {name}")
        axes[0].axis("off")

        axes[1].imshow(restored, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Restored")
        axes[1].axis("off")

        fig.suptitle(f"Sample {state['idx'] + 1} / {n}   (press ← or → to browse)")
        fig.canvas.draw_idle()

    def next_image(event=None):
        state["idx"] = (state["idx"] + 1) % n
        show_current()

    def prev_image(event=None):
        state["idx"] = (state["idx"] - 1) % n
        show_current()

    def on_key(event):
        if event.key == "right":
            next_image()
        elif event.key == "left":
            prev_image()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # on-screen buttons too, in case keyboard focus is elsewhere
    ax_prev = plt.axes([0.35, 0.02, 0.1, 0.06])
    ax_next = plt.axes([0.55, 0.02, 0.1, 0.06])
    btn_prev = Button(ax_prev, "← Prev")
    btn_next = Button(ax_next, "Next →")
    btn_prev.on_clicked(prev_image)
    btn_next.on_clicked(next_image)

    show_current()
    plt.show()


if __name__ == "__main__":
    main()
