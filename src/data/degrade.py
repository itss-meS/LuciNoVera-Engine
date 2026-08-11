import numpy as np
import cv2


def add_speckle_noise(img, sigma_range=(0.05, 0.3)):
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return img + img * noise


def add_gaussian_degradation(img, blur_kernel_range=(3, 7), noise_std_range=(0.01, 0.08)):
    k = np.random.choice(range(blur_kernel_range[0], blur_kernel_range[1] + 1, 2))
    blurred = cv2.GaussianBlur(img, (int(k), int(k)), 0)
    noise_std = np.random.uniform(*noise_std_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * noise_std
    return blurred + noise


def random_resize(img, scale_choices=(0.5, 0.25)):
    scale = np.random.choice(scale_choices)
    interp = np.random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_AREA])
    h, w = img.shape
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)
    return cv2.resize(small, (w, h), interpolation=interp)


def random_degrade(img):
    """Applies a random combination of the degradations above to an already
    somewhat-degraded input, widening the range of degradations seen during
    training so the model generalizes better to OOD test sources."""
    ops = []
    if np.random.rand() < 0.7:
        ops.append(add_speckle_noise)
    if np.random.rand() < 0.5:
        ops.append(add_gaussian_degradation)
    if np.random.rand() < 0.3:
        ops.append(random_resize)

    np.random.shuffle(ops)
    out = img.copy()
    for op in ops:
        out = op(out)
    return out.astype(np.float32)
