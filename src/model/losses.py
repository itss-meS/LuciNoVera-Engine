import torch
import torch.nn as nn
import torch.nn.functional as F


class BrightnessConsistencyLoss(nn.Module):
    """Directly penalizes per-image mean and std mismatch between prediction
    and target. Fixes systematic brightness/contrast bias at the source,
    rather than trying to correct it after the fact at inference time."""

    def forward(self, pred, target):
        pred_mean = pred.mean(dim=[1, 2, 3])
        target_mean = target.mean(dim=[1, 2, 3])
        pred_std = pred.std(dim=[1, 2, 3])
        target_std = target.std(dim=[1, 2, 3])
        return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def gaussian_window(size, sigma, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", torch.zeros(1))
        self._built = False

    def _build(self, device):
        self.window = gaussian_window(self.window_size, self.sigma, device)
        self._built = True

    def forward(self, pred, target, data_range=1.0):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target = pred.float(), target.float()
            if not self._built or self.window.device != pred.device:
                self._build(pred.device)
            window = self.window
            pad = self.window_size // 2

            mu_p = F.conv2d(pred, window, padding=pad)
            mu_t = F.conv2d(target, window, padding=pad)
            mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

            sigma_p2 = (F.conv2d(pred * pred, window, padding=pad) - mu_p2).clamp(min=0)
            sigma_t2 = (F.conv2d(target * target, window, padding=pad) - mu_t2).clamp(min=0)
            sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

            c1 = (0.01 * data_range) ** 2
            c2 = (0.03 * data_range) ** 2
            numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
            denominator = (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
            ssim_map = numerator / denominator.clamp(min=1e-8)
            return 1 - ssim_map.mean()


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _sobel(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred, target):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return F.l1_loss(self._sobel(pred.float()), self._sobel(target.float()))


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_f = torch.fft.rfft2(pred.float(), norm="ortho")
            target_f = torch.fft.rfft2(target.float(), norm="ortho")
            return F.l1_loss(torch.abs(pred_f), torch.abs(target_f))


class UncertaintyLoss(nn.Module):
    """Heteroscedastic loss: confidence map scales the per-pixel error term.
    Runs in float32 even under AMP since 1/confidence can overflow fp16 when
    confidence is near zero."""

    def forward(self, pred, target, confidence):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target, confidence = pred.float(), target.float(), confidence.float()
            confidence = torch.clamp(confidence, min=1e-3)
            precision = 1.0 / confidence
            error = (pred - target) ** 2
            return torch.mean(precision * error + torch.log(confidence))


class LaplacianLoss(nn.Module):
    """Multi-scale Laplacian pyramid loss: penalizes lost high-frequency detail
    at 3 scales. Directly targets the 'blur' failure mode where a model
    outputs a safe, smoothed average instead of committing to fine texture."""

    def _lap(self, img):
        blur = F.avg_pool2d(img, 3, stride=1, padding=1)
        return img - blur

    def forward(self, pred, target):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target = pred.float(), target.float()
            loss = 0.0
            p, t = pred, target
            for _ in range(3):
                loss = loss + F.l1_loss(self._lap(p), self._lap(t))
                if min(p.shape[-2:]) < 4:
                    break
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
            return loss


class PerceptualLoss(nn.Module):
    """Compares VGG features at multiple depths instead of raw pixels.
    Shallow layers (relu1_2, relu2_2) respond to fine texture and edges —
    the main anti-blur signal. Deeper layers (relu3_3) capture broader
    structure/content. Using only a deep layer, as a single-layer perceptual
    loss does, under-penalizes blur since deep features tolerate some
    softness; combining shallow + deep gives a much stronger sharpness push."""

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:16]
        self.layer_weights = [1.0, 1.0, 0.5]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x):
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return [f1, f2, f3]

    def forward(self, pred, target):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target = pred.float(), target.float()
            pred3 = pred.repeat(1, 3, 1, 1)
            target3 = target.repeat(1, 3, 1, 1)
            pred3 = (pred3 - self.mean) / self.std
            target3 = (target3 - self.mean) / self.std
            pred_feats = self._features(pred3)
            with torch.no_grad():
                target_feats = self._features(target3)
            loss = 0.0
            for pf, tf, w in zip(pred_feats, target_feats, self.layer_weights):
                loss = loss + w * F.l1_loss(pf, tf)
            return loss


class CombinedLoss(nn.Module):
    def __init__(self, w_charbonnier=1.0, w_ssim=0.2, w_edge=0.1, w_fft=0.05,
                 w_uncertainty=0.05, w_laplacian=0.0, w_perceptual=0.0, w_brightness=0.0):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.fft = FFTLoss()
        self.uncertainty = UncertaintyLoss()
        self.laplacian = LaplacianLoss()
        self.brightness = BrightnessConsistencyLoss()
        self.use_perceptual = w_perceptual > 0
        if self.use_perceptual:
            self.perceptual = PerceptualLoss()
        self.weights = dict(charbonnier=w_charbonnier, ssim=w_ssim, edge=w_edge,
                             fft=w_fft, uncertainty=w_uncertainty, laplacian=w_laplacian,
                             perceptual=w_perceptual, brightness=w_brightness)

    def forward(self, pred, target, confidence):
        l_charb = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred, target)
        l_edge = self.edge(pred, target)
        l_fft = self.fft(pred, target)
        l_unc = self.uncertainty(pred, target, confidence)
        l_lap = self.laplacian(pred, target)
        l_bright = self.brightness(pred, target)

        total = (self.weights["charbonnier"] * l_charb +
                 self.weights["ssim"] * l_ssim +
                 self.weights["edge"] * l_edge +
                 self.weights["fft"] * l_fft +
                 self.weights["uncertainty"] * l_unc +
                 self.weights["laplacian"] * l_lap +
                 self.weights["brightness"] * l_bright)

        parts = dict(charbonnier=l_charb.item(), ssim=l_ssim.item(), edge=l_edge.item(),
                     fft=l_fft.item(), uncertainty=l_unc.item(), laplacian=l_lap.item(),
                     brightness=l_bright.item())

        if self.use_perceptual:
            l_perc = self.perceptual(pred, target)
            total = total + self.weights["perceptual"] * l_perc
            parts["perceptual"] = l_perc.item()

        parts["total"] = total.item()

        if not torch.isfinite(total):
            bad = [k for k, v in parts.items() if k != "total" and not (v == v and abs(v) != float("inf"))]
            parts["bad_terms"] = bad

        return total, parts
