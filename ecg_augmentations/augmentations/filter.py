import torch

from ..base import Transform, as_range, uniform
from ..ops import butter_gain, fft_filter


class FrequencyFilter(Transform):
    """Zero-phase Butterworth-shaped filter with a random cutoff (Hz)."""

    highpass = False

    def __init__(self, fs, cutoff, order=4, p=1.0):
        super().__init__(p)
        self.fs = fs
        self.cutoff = as_range(cutoff, "cutoff", low=1e-6, high=fs / 2 - 1e-6)
        self.order = order

    def apply(self, x, generator=None):
        cutoff = uniform(self.cutoff, (x.shape[0], 1, 1), x, generator)
        return fft_filter(x, self.fs, butter_gain(cutoff, self.order, self.highpass))


class HighPassFilter(FrequencyFilter):
    """High-pass filter; removes baseline drift and, above ~0.5 Hz, distorts ST segments.

    Args:
        fs: sampling rate (Hz).
        cutoff: cutoff or (low, high) range in Hz.
        order: steepness of the Butterworth-shaped response.
    """

    highpass = True

    def __init__(self, fs, cutoff=(0.05, 1.0), order=4, p=1.0):
        super().__init__(fs, cutoff, order, p)


class LowPassFilter(FrequencyFilter):
    """Low-pass filter; removes high-frequency noise and blunts sharp QRS features.

    Args:
        fs: sampling rate (Hz).
        cutoff: cutoff or (low, high) range in Hz, below fs / 2.
        order: steepness of the Butterworth-shaped response.
    """

    def __init__(self, fs, cutoff=(20.0, 45.0), order=4, p=1.0):
        super().__init__(fs, cutoff, order, p)


class FrequencyMask(Transform):
    """Remove random frequency bands, shared across leads.

    Args:
        fs: sampling rate (Hz).
        width: band width or (low, high) range in Hz.
        n_masks: number of bands.
        max_freq: highest frequency (Hz) a band may reach; defaults to fs / 2.
    """

    def __init__(self, fs, width=(1.0, 5.0), n_masks=1, max_freq=None, p=1.0):
        super().__init__(p)
        self.fs = fs
        self.max_freq = fs / 2 if max_freq is None else max_freq
        self.width = as_range(width, "width", low=0, high=self.max_freq)
        self.n_masks = n_masks

    def apply(self, x, generator=None):
        b, _, t = x.shape
        spectrum = torch.fft.rfft(x)
        freqs = torch.fft.rfftfreq(t, d=1 / self.fs).to(x.device, x.dtype)
        keep = torch.ones(b, 1, len(freqs), dtype=torch.bool, device=x.device)
        for _ in range(self.n_masks):
            width = uniform(self.width, (b, 1, 1), x, generator)
            start = uniform((0, 1), (b, 1, 1), x, generator) * (self.max_freq - width)
            keep &= (freqs < start) | (freqs >= start + width)
        return torch.fft.irfft(spectrum * keep, n=t)
