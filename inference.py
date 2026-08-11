import argparse
import os
import time
import numpy as np
import torch
import cv2

from src.data.npy_dataset import InferenceNpyDataset
from src.model.restoration_net import RestorationNet
from src.utils.tta import tta_forward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--weights", default="weights/restoration_model_final.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--output_dtype", default="float32", choices=["float32", "uint16", "uint8"])
    p.add_argument("--unsharp", action="store_true",
                    help="classical post-process sharpening, not learned by the model — use conservatively")
    p.add_argument("--unsharp_sigma", type=float, default=1.0)
    p.add_argument("--unsharp_strength", type=float, default=0.4,
                    help="0.2-0.5 recommended; higher risks ringing/halo artifacts")
    return p.parse_args()


def unsharp_mask(img, sigma=1.0, strength=0.4):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = img + strength * (img - blurred)
    return np.clip(sharpened, img.min(), img.max())


def cast_output(arr, dtype):
    if dtype == "float32":
        return arr.astype(np.float32)
    if dtype == "uint16":
        return np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)
    if dtype == "uint8":
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(dtype)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: --device {args.device} requested but CUDA is not available in this "
              f"Python/torch environment. Falling back to CPU (inference will be slower). "
              f"If this is unexpected, this machine's torch install is likely CPU-only — "
              f"see README 'Environment Setup' for installing a CUDA-enabled build.")
        args.device = "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    t_start = time.time()
    checkpoint = torch.load(args.weights, map_location=args.device)
    model = RestorationNet(config=checkpoint.get("config", "medium"),
                            scale=checkpoint.get("scale", 2)).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = InferenceNpyDataset(args.input_dir)
    print(f"Found {len(dataset)} input files. Model init took {time.time() - t_start:.3f}s")

    per_image_times = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            x = sample["degraded"].unsqueeze(0).to(args.device)

            t0 = time.time()
            if args.tta:
                restored, _ = tta_forward(model, x)
            else:
                restored, _ = model(x)
            torch.cuda.synchronize() if args.device.startswith("cuda") else None
            elapsed = time.time() - t0
            per_image_times.append(elapsed)

            restored_np = restored.clamp(0, 1).squeeze().cpu().numpy()
            if args.unsharp:
                restored_np = unsharp_mask(restored_np, sigma=args.unsharp_sigma, strength=args.unsharp_strength)
            restored_np = cast_output(restored_np, args.output_dtype)

            out_path = os.path.join(args.output_dir, sample["name"])
            np.save(out_path, restored_np)
            print(f"[{i + 1}/{len(dataset)}] {sample['name']} -> {elapsed * 1000:.1f} ms")

    total_time = time.time() - t_start
    avg_time = float(np.mean(per_image_times)) if per_image_times else 0.0
    print(f"Done. Total time: {total_time:.3f}s | Avg inference time/image: {avg_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
