import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

_lpips_model = None


def compute_psnr(pred, target, data_range=1.0):
    return sk_psnr(target, pred, data_range=data_range)


def compute_ssim(pred, target, data_range=1.0):
    return sk_ssim(target, pred, data_range=data_range)


def compute_lpips(pred, target, device="cpu"):
    """Lazily loads LPIPS (AlexNet backbone) on first call. Expects pred/target
    as float32 (H, W) arrays in [0, 1]."""
    global _lpips_model
    import torch
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()

    def to_tensor(x):
        t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        t = t.repeat(1, 3, 1, 1) * 2 - 1
        return t.to(device)

    with torch.no_grad():
        d = _lpips_model(to_tensor(pred), to_tensor(target))
    return float(d.item())


def evaluate_batch(pred, target, device="cpu", with_lpips=True):
    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)
    result = {"psnr": psnr, "ssim": ssim}
    if with_lpips:
        result["lpips"] = compute_lpips(pred, target, device=device)
    return result
