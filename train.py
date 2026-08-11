import argparse
import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from src.data.npy_dataset import PairedNpyDataset
from src.model.restoration_net import RestorationNet
from src.model.losses import CombinedLoss
from src.utils.metrics import compute_psnr, compute_ssim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_gt_dir", required=True)
    p.add_argument("--train_degraded_dir", required=True)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--config", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="resume from output_dir/last_checkpoint.pt if present")
    p.add_argument("--amp", action="store_true", help="mixed-precision training, cuda only, ~1.5-2x faster")
    p.add_argument("--init_weights", default=None,
                    help="load model weights only (e.g. restoration_model_final.pt) and start a fresh "
                         "optimizer/scheduler at epoch 0 — for fine-tuning with new loss weights or LR, "
                         "as opposed to --resume which continues the exact same training state")
    p.add_argument("--w_charbonnier", type=float, default=1.0)
    p.add_argument("--w_ssim", type=float, default=0.2)
    p.add_argument("--w_edge", type=float, default=0.1)
    p.add_argument("--w_fft", type=float, default=0.05)
    p.add_argument("--w_uncertainty", type=float, default=0.05)
    p.add_argument("--w_laplacian", type=float, default=0.0)
    p.add_argument("--w_perceptual", type=float, default=0.0,
                    help="VGG feature-space loss weight; the strongest lever for real sharpness, try 0.3-0.5")
    p.add_argument("--w_brightness", type=float, default=0.0,
                    help="directly penalizes mean/contrast bias between prediction and target")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: --device {args.device} requested but CUDA is not available in this "
              f"Python/torch environment. Falling back to CPU (training will be much slower). "
              f"If this is unexpected, this machine's torch install is likely CPU-only — "
              f"see README 'Environment Setup' for installing a CUDA-enabled build.")
        args.device = "cpu"
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    writer = SummaryWriter(log_dir)

    full_dataset = PairedNpyDataset(
        args.train_gt_dir, args.train_degraded_dir,
        patch_size=args.patch_size, train=True, synth_augment=True,
    )
    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True,
                               persistent_workers=(args.num_workers > 0),
                               pin_memory=(args.device.startswith("cuda")))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    sample = full_dataset[0]
    scale = sample["scale"]

    model = RestorationNet(config=args.config, scale=scale).to(args.device)
    print(f"Model params: {model.count_params() / 1e6:.2f}M")
    print(f"Train pairs: {len(train_set)} | Val pairs: {len(val_set)} | "
          f"Batches/epoch: {len(train_loader)} | Batch size: {args.batch_size}")

    criterion = CombinedLoss(
        w_charbonnier=args.w_charbonnier, w_ssim=args.w_ssim, w_edge=args.w_edge,
        w_fft=args.w_fft, w_uncertainty=args.w_uncertainty, w_laplacian=args.w_laplacian,
        w_perceptual=args.w_perceptual, w_brightness=args.w_brightness,
    ).to(args.device)
    print(f"Loss weights: charbonnier={args.w_charbonnier} ssim={args.w_ssim} edge={args.w_edge} "
          f"fft={args.w_fft} uncertainty={args.w_uncertainty} laplacian={args.w_laplacian} "
          f"perceptual={args.w_perceptual} brightness={args.w_brightness}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.amp and not use_amp:
        print("--amp requested but device is not cuda; running in full precision")

    best_ssim = -1.0
    global_step = 0
    start_epoch = 0

    if args.init_weights:
        if args.resume:
            raise ValueError("--init_weights and --resume are mutually exclusive")
        init_ckpt = torch.load(args.init_weights, map_location=args.device)
        model.load_state_dict(init_ckpt["model_state"])
        print(f"loaded model weights from {args.init_weights} for fine-tuning "
              f"(fresh optimizer/scheduler, starting at epoch 1)")

    last_ckpt_path = os.path.join(args.output_dir, "last_checkpoint.pt")
    if args.resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if use_amp and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_ssim = ckpt["best_ssim"]
        global_step = ckpt["global_step"]
        print(f"resumed from {last_ckpt_path} at epoch {start_epoch + 1}, best_ssim so far {best_ssim:.4f}")
    elif args.resume:
        print(f"--resume was set but {last_ckpt_path} does not exist; starting fresh")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        n_valid_batches = 0
        consecutive_bad = 0
        for batch in train_loader:
            degraded = batch["degraded"].to(args.device)
            gt = batch["gt"].to(args.device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                restored, confidence = model(degraded)
                loss, parts = criterion(restored, gt, confidence)

            if not torch.isfinite(loss):
                consecutive_bad += 1
                bad_terms = parts.get("bad_terms", ["unknown"])
                print(f"  WARNING: non-finite loss at step {global_step}, bad term(s): {bad_terms}, "
                      f"skipping this batch ({consecutive_bad} consecutive)")
                global_step += 1
                if consecutive_bad >= 50:
                    raise RuntimeError(
                        "50 consecutive non-finite batches — training has diverged. "
                        "Stopping now so last_checkpoint.pt is NOT overwritten with bad weights. "
                        "Inspect loss weights / learning rate before resuming."
                    )
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            consecutive_bad = 0

            epoch_loss += parts["total"]
            n_valid_batches += 1
            writer.add_scalar("train/loss", parts["total"], global_step)
            global_step += 1

        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(
                f"Model weights contain NaN/Inf after epoch {epoch + 1}. "
                "Stopping now so last_checkpoint.pt is NOT overwritten with corrupted weights. "
                "Resume from the previous checkpoint after investigating."
            )

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_valid_batches)
        epoch_time = time.time() - epoch_start
        epochs_left = args.epochs - (epoch + 1)
        eta_seconds = epoch_time * epochs_left
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        print(f"epoch {epoch + 1}/{args.epochs} - train_loss {avg_loss:.4f} - "
              f"{epoch_time:.1f}s/epoch - ETA {eta_str}")

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_psnr, val_ssim = validate(model, val_loader, args.device)
            writer.add_scalar("val/psnr", val_psnr, epoch)
            writer.add_scalar("val/ssim", val_ssim, epoch)
            print(f"  val_psnr {val_psnr:.3f} val_ssim {val_ssim:.4f}")

            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save({"model_state": model.state_dict(), "config": args.config,
                            "scale": scale}, os.path.join(args.output_dir, "restoration_model_final.pt"))
                print("  saved new best checkpoint")

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_ssim": best_ssim,
            "global_step": global_step,
            "config": args.config,
            "scale": scale,
        }, last_ckpt_path)

    writer.close()


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    psnrs, ssims = [], []
    for batch in val_loader:
        degraded = batch["degraded"].to(device)
        gt = batch["gt"].to(device)
        restored, _ = model(degraded)
        pred_np = restored.clamp(0, 1).squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()
        psnrs.append(compute_psnr(pred_np, gt_np))
        ssims.append(compute_ssim(pred_np, gt_np))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


if __name__ == "__main__":
    main()
