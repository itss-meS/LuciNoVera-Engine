import torch
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/last_checkpoint.pt"
ckpt = torch.load(path, map_location="cpu")
state = ckpt["model_state"]

has_nan = False
for name, tensor in state.items():
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN/Inf found in: {name}")
        has_nan = True

if has_nan:
    print(f"\n{path} is CORRUPTED (contains NaN/Inf weights).")
else:
    print(f"\n{path} is CLEAN (no NaN/Inf) — weights just stopped updating, not corrupted.")

if "epoch" in ckpt:
    print(f"Checkpoint epoch: {ckpt['epoch']}")
if "best_ssim" in ckpt:
    print(f"best_ssim recorded: {ckpt['best_ssim']}")
