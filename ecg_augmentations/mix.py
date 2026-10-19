import torch

from .base import randint, randperm, uniform


def beta(alpha, n, like, generator=None):
    """Sample Beta(alpha, alpha) as a ratio of Gamma draws, honoring the generator."""
    device = generator.device if generator is not None else like.device
    concentration = torch.full((2, n), float(alpha), dtype=torch.float64, device=device)
    gamma = torch._standard_gamma(concentration, generator=generator)
    return (gamma[0] / gamma.sum(0)).to(like.device, like.dtype)


class BatchMix(torch.nn.Module):
    """Base for label-mixing batch augmentations: forward(x, y) -> (x, y).

    x has shape (B, ..., T); y holds float targets of shape (B, ...), e.g. one-hot.
    """

    def __init__(self, alpha):
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha

    def forward(self, x, y, generator=None):
        if not y.is_floating_point():
            raise TypeError("labels must be float, e.g. one-hot encoded")
        lam = beta(self.alpha, x.shape[0], x, generator)
        perm = randperm(x.shape[0], generator).to(x.device)
        x, lam = self.mix(x, x[perm], lam, generator)
        lam = lam.to(y).reshape(-1, *[1] * (y.ndim - 1))
        return x, lam * y + (1 - lam) * y[perm]

    def extra_repr(self):
        return f"alpha={self.alpha}"


class Mixup(BatchMix):
    """Blend each sample with another sample of the batch, and their labels."""

    def __init__(self, alpha=0.2):
        super().__init__(alpha)

    def mix(self, x, other, lam, generator=None):
        weight = lam.reshape(-1, *[1] * (x.ndim - 1))
        return weight * x + (1 - weight) * other, lam


class CutMix(BatchMix):
    """Replace a random time window with the same window of another sample.

    Labels are mixed by the fraction of samples kept. Splicing can create
    implausible rhythms, so it suits morphology-level more than rhythm labels.
    """

    def __init__(self, alpha=1.0):
        super().__init__(alpha)

    def mix(self, x, other, lam, generator=None):
        t = x.shape[-1]
        shape = (-1, *[1] * (x.ndim - 1))
        length = ((1 - lam) * t).round().reshape(shape)
        start = randint(0, t - length + 1, length.shape, x, generator)
        idx = torch.arange(t, device=x.device)
        window = (idx >= start) & (idx < start + length)
        return torch.where(window, other, x), 1 - length.flatten() / t
