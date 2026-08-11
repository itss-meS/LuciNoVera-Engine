import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.data.npy_dataset import PairedNpyDataset
from src.model.restoration_net import RestorationNet
from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_gt_dir", required=True)
    p.add_argument("--train_degraded_dir", required=True)
    p.add_argument("--weights", default="weights/restoration_model_final.pt")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--no_lpips", action="store_true", help="skip LPIPS (slower, needs extra download)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: --device {args.device} requested but CUDA is not available in this "
              f"Python/torch environment. Falling back to CPU (evaluation will be slower). "
              f"If this is unexpected, this machine's torch install is likely CPU-only — "
              f"see README 'Environment Setup' for installing a CUDA-enabled build.")
        args.device = "cpu"

    full_dataset = PairedNpyDataset(
        args.train_gt_dir, args.train_degraded_dir,
        patch_size=None, train=False, synth_augment=False,
    )
    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    _, val_set = random_split(full_dataset, [n_train, n_val],
                               generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(val_set)} held-out validation pairs "
          f"(same split train.py used, seed={args.seed})")

    checkpoint = torch.load(args.weights, map_location=args.device)
    model = RestorationNet(config=checkpoint["config"], scale=checkpoint["scale"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    from src.utils.tta import tta_forward

    psnrs, ssims, lpipss, times = [], [], [], []
    pred_means, gt_means, pred_sharpness, gt_sharpness = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            degraded = batch["degraded"].to(args.device)
            gt = batch["gt"].to(args.device)

            t0 = time.time()
            if args.tta:
                restored, _ = tta_forward(model, degraded)
            else:
                restored, _ = model(degraded)
            elapsed = time.time() - t0
            times.append(elapsed)

            pred_np = restored.clamp(0, 1).squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            pred_means.append(float(pred_np.mean()))
            gt_means.append(float(gt_np.mean()))
            pred_sharpness.append(float(laplacian_variance(pred_np)))
            gt_sharpness.append(float(laplacian_variance(gt_np)))

            p = compute_psnr(pred_np, gt_np)
            s = compute_ssim(pred_np, gt_np)
            psnrs.append(p)
            ssims.append(s)

            if (not args.no_lpips):
                l = compute_lpips(pred_np, gt_np, device=args.device)
                lpipss.append(l)
                print(f"[{i + 1}/{len(val_set)}] {batch['name'][0]}  "
                      f"PSNR={p:.2f}  SSIM={s:.4f}  LPIPS={l:.4f}  time={elapsed * 1000:.1f}ms")
            else:
                print(f"[{i + 1}/{len(val_set)}] {batch['name'][0]}  "
                      f"PSNR={p:.2f}  SSIM={s:.4f}  time={elapsed * 1000:.1f}ms")

    print("\n===== FINAL RESULTS =====")
    print(f"Avg PSNR:  {np.mean(psnrs):.2f} dB")
    print(f"Avg SSIM:  {np.mean(ssims):.4f}")
    if (not args.no_lpips):
        print(f"Avg LPIPS: {np.mean(lpipss):.4f}")
    print(f"Avg inference time: {np.mean(times) * 1000:.2f} ms/image")

    print("\n===== BRIGHTNESS / SHARPNESS DIAGNOSTIC (raw model output vs real GT) =====")
    pm, gm = np.mean(pred_means), np.mean(gt_means)
    ps, gs = np.mean(pred_sharpness), np.mean(gt_sharpness)
    print(f"Mean brightness — restored: {pm:.4f}  |  ground truth: {gm:.4f}  |  bias: {pm - gm:+.4f}")
    print(f"Sharpness (Laplacian var) — restored: {ps:.4f}  |  ground truth: {gs:.4f}  |  "
          f"ratio: {ps / max(gs, 1e-8):.2f}")
    if abs(pm - gm) > 0.03:
        print("-> Model's raw output has a real, systematic brightness bias vs ground truth.")
        print("   This is a training-time issue, not fixable by rescaling predict.py's output.")
    else:
        print("-> No significant brightness bias in the model's raw predictions.")
    if ps / max(gs, 1e-8) < 0.5:
        print("-> Model output is significantly less sharp than ground truth (real blur, confirmed).")


def laplacian_variance(img):
    import cv2
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)
    return lap.var()


if __name__ == "__main__":
    main()
