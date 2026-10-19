import torch

from ..base import Transform, randint


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
