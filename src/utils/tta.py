import torch


def _transform(x, mode):
    if mode == "identity":
        return x
    if mode == "hflip":
        return torch.flip(x, dims=[3])
    if mode == "vflip":
        return torch.flip(x, dims=[2])
    if mode == "rot90":
        return torch.rot90(x, 1, dims=[2, 3])
    raise ValueError(mode)


def _inverse_transform(x, mode):
    if mode == "identity":
        return x
    if mode == "hflip":
        return torch.flip(x, dims=[3])
    if mode == "vflip":
        return torch.flip(x, dims=[2])
    if mode == "rot90":
        return torch.rot90(x, -1, dims=[2, 3])
    raise ValueError(mode)


MODES = ["identity", "hflip", "vflip", "rot90"]


@torch.no_grad()
def tta_forward(model, x):
    """Runs the model on flip/rotate variants of x and averages the results
    back in the original orientation. Confidence maps are averaged too."""
    restored_sum, conf_sum = None, None
    for mode in MODES:
        xt = _transform(x, mode)
        restored, confidence = model(xt)
        restored = _inverse_transform(restored, mode)
        confidence = _inverse_transform(confidence, mode)
        restored_sum = restored if restored_sum is None else restored_sum + restored
        conf_sum = confidence if conf_sum is None else conf_sum + confidence
    n = len(MODES)
    return restored_sum / n, conf_sum / n
