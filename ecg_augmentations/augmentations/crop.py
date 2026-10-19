import torch

from ..base import Transform, as_range, randint, uniform
from ..ops import interp


def crop_length(n_samples, duration, fs):
    if (n_samples is None) == (duration is None):
        raise ValueError("give exactly one of n_samples or duration")
    if duration is not None:
        if fs is None:
            raise ValueError("duration requires fs")
        n_samples = round(duration * fs)
    if n_samples < 1:
        raise ValueError(f"crop length must be positive, got {n_samples}")
    return int(n_samples)


class RandomCrop(Transform):
    """Crop a random window of fixed length, shared across leads.

    Args:
        n_samples: window length in samples, or
        duration: window length in seconds, together with fs (Hz).
    """

    def __init__(self, n_samples=None, duration=None, fs=None):
        super().__init__()
        self.n_samples = crop_length(n_samples, duration, fs)

    def apply(self, x, generator=None):
        b, c, t = x.shape
        if self.n_samples > t:
            raise ValueError(
                f"cannot crop {self.n_samples} samples from a signal of length {t}"
            )
        start = randint(0, t - self.n_samples + 1, (b, 1, 1), x, generator)
        idx = start + torch.arange(self.n_samples, device=x.device)
        return x.gather(-1, idx.expand(b, c, -1))


class RandomResizedCrop(Transform):
    """Crop a random window and resample it to a fixed length, shared across leads.

    Resampling changes the time scale (heart rate and intervals) by n_samples / window.

    Args:
        n_samples: output length in samples.
        scale: window length as a fraction, or (low, high) range of fractions, of the input.
    """

    def __init__(self, n_samples, scale=(0.5, 1.0)):
        super().__init__()
        self.n_samples = crop_length(n_samples, None, None)
        self.scale = as_range(scale, "scale", low=1e-6, high=1)

    def apply(self, x, generator=None):
        b, _, t = x.shape
        length = (uniform(self.scale, (b, 1, 1), x, generator) * t).clamp(min=1)
        start = uniform((0, 1), (b, 1, 1), x, generator) * (t - length)
        step = (length - 1) / max(self.n_samples - 1, 1)
        idx = torch.arange(self.n_samples, device=x.device, dtype=x.dtype)
        return interp(x, start + idx * step)
