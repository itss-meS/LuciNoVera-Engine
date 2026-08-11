import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.degrade import random_degrade


def load_npy_as_hw(path, mmap=True):
    """Loads a .npy file and returns a 2D (H, W) float32 array regardless of
    whether the stored array is (H,W), (H,W,1), or (1,H,W)."""
    arr = np.load(path, mmap_mode="r" if mmap else None)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unexpected 3D shape {arr.shape} in {path}; expected a singleton channel dim.")
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected ndim {arr.ndim} for {path}; expected (H,W), (H,W,1) or (1,H,W).")
    return arr.astype(np.float32)


def normalize_per_sample(arr):
    """Min-max normalizes to [0, 1] using this sample's own range. Returns
    (normalized_array, min_val, max_val) so the transform can be inverted."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr), lo, hi
    norm = (arr - lo) / (hi - lo)
    return norm, lo, hi


def denormalize(arr, lo, hi):
    if hi - lo < 1e-8:
        return np.full_like(arr, lo)
    return arr * (hi - lo) + lo


class PairedNpyDataset(Dataset):
    """Pairs degraded/ground-truth .npy files by matching filename across two
    directories. Applies additional synthetic degradation on top of the
    provided degraded image during training to broaden the degradation
    distribution the model sees (see data/degrade.py)."""

    def __init__(self, gt_dir, degraded_dir, patch_size=128, train=True, synth_augment=True):
        self.gt_dir = gt_dir
        self.degraded_dir = degraded_dir
        self.patch_size = patch_size
        self.train = train
        self.synth_augment = synth_augment

        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        pairs = []
        for gt_path in gt_files:
            name = os.path.basename(gt_path)
            deg_path = os.path.join(degraded_dir, name)
            if os.path.exists(deg_path):
                pairs.append((deg_path, gt_path))
        if not pairs:
            raise RuntimeError(
                f"No matching filename pairs found between {gt_dir} and {degraded_dir}. "
                "Check pairing convention (filename match expected) before training."
            )
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _augment_geometry(self, deg, gt, scale):
        if np.random.rand() < 0.5:
            deg, gt = deg[:, ::-1].copy(), gt[:, ::-1].copy()
        if np.random.rand() < 0.5:
            deg, gt = deg[::-1, :].copy(), gt[::-1, :].copy()
        k = np.random.randint(4)
        if k:
            deg, gt = np.rot90(deg, k).copy(), np.rot90(gt, k).copy()
        return deg, gt

    def _crop(self, deg, gt, scale):
        dh, dw = deg.shape
        p = min(self.patch_size, dh, dw)
        y = np.random.randint(0, dh - p + 1) if dh > p else 0
        x = np.random.randint(0, dw - p + 1) if dw > p else 0
        deg_crop = deg[y:y + p, x:x + p]
        gt_crop = gt[y * scale:(y + p) * scale, x * scale:(x + p) * scale]
        return deg_crop, gt_crop

    def __getitem__(self, idx):
        deg_path, gt_path = self.pairs[idx]
        degraded = load_npy_as_hw(deg_path)
        gt = load_npy_as_hw(gt_path)

        scale = round(gt.shape[0] / degraded.shape[0])
        assert scale in (1, 2, 4), f"Unsupported scale factor {scale} for {deg_path}"

        if self.synth_augment and self.train:
            degraded = random_degrade(degraded)

        if self.train and self.patch_size:
            degraded, gt = self._crop(degraded, gt, scale)
            degraded, gt = self._augment_geometry(degraded, gt, scale)

        degraded_norm, d_lo, d_hi = normalize_per_sample(degraded)
        gt_norm, g_lo, g_hi = normalize_per_sample(gt)

        degraded_t = torch.from_numpy(degraded_norm).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_norm).unsqueeze(0).float()

        return {
            "degraded": degraded_t,
            "gt": gt_t,
            "scale": scale,
            "d_lo": d_lo, "d_hi": d_hi,
            "g_lo": g_lo, "g_hi": g_hi,
            "name": os.path.basename(deg_path),
        }


class InferenceNpyDataset(Dataset):
    """Loads standalone degraded .npy files for inference (no ground truth)."""

    def __init__(self, input_dir):
        self.files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
        if not self.files:
            raise RuntimeError(f"No .npy files found in {input_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = load_npy_as_hw(path, mmap=False)
        norm, lo, hi = normalize_per_sample(arr)
        p_lo, p_hi = np.percentile(arr, 1), np.percentile(arr, 99)
        clipped = np.clip(arr, p_lo, p_hi)
        r_mean, r_std = float(clipped.mean()), float(clipped.std())
        tensor = torch.from_numpy(norm).unsqueeze(0).float()
        return {
            "degraded": tensor,
            "lo": lo, "hi": hi,
            "p_lo": float(p_lo), "p_hi": float(p_hi),
            "r_mean": r_mean, "r_std": r_std,
            "name": os.path.basename(path),
            "orig_shape": arr.shape,
        }
