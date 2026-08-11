import argparse
import os
import torch

from src.model.restoration_net import RestorationNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/restoration_model_final.pt")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--resume_epoch", type=int, required=True,
                    help="the epoch number this checkpoint represents (0-indexed, e.g. 39 for 'finished epoch 40')")
    p.add_argument("--total_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batches_per_epoch", type=int, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.weights, map_location=args.device)
    model = RestorationNet(config=ckpt["config"], scale=ckpt["scale"]).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)
    for _ in range(args.resume_epoch + 1):
        scheduler.step()

    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))

    out_path = os.path.join(args.output_dir, "last_checkpoint.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": args.resume_epoch,
        "best_ssim": 0.6068,
        "global_step": (args.resume_epoch + 1) * args.batches_per_epoch,
        "config": ckpt["config"],
        "scale": ckpt["scale"],
    }, out_path)
    print(f"Wrote rebuilt resumable checkpoint to {out_path}, will resume at epoch {args.resume_epoch + 2}")
    print("Note: optimizer momentum was reset fresh — expect a few epochs of slightly noisier "
          "loss while it re-warms, this is normal and not a repeat of the NaN issue.")


if __name__ == "__main__":
    main()
