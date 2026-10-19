import math, torch

from ..base import Transform, as_range, randn, uniform
from ..ops import add_noise


def time_axis(x, fs):
    return torch.arange(x.shape[-1], device=x.device, dtype=x.dtype) / fs


class GaussianNoise(Transform):
    """Add white Gaussian noise, independent per lead.

    Args:
        snr_db: signal-to-noise ratio or (low, high) range in dB, per lead.
    """

    def __init__(self, snr_db=(15.0, 35.0), p=1.0):
        super().__init__(p)
        self.snr_db = as_range(snr_db, "snr_db")

    def apply(self, x, generator=None):
        snr = uniform(self.snr_db, (x.shape[0], 1, 1), x, generator)
        return add_noise(x, randn(x.shape, x, generator), snr)


class BaselineWander(Transform):
    """Add low-frequency sinusoidal drift, with random phase and amplitude per lead.

    Args:
        fs: sampling rate (Hz).
        amplitude: peak amplitude or (low, high) range; relative to each lead's std
            if `relative`, else in signal units (e.g. mV).
        freq: drift frequency or (low, high) range in Hz, shared across leads.
        n_components: number of summed sinusoids.
    """

    def __init__(
        self,
        fs,
        amplitude=(0.05, 0.5),
        freq=(0.05, 0.5),
        n_components=2,
        relative=True,
        p=1.0,
    ):
        super().__init__(p)
        self.fs = fs
        self.amplitude = as_range(amplitude, "amplitude", low=0)
        self.freq = as_range(freq, "freq", low=0, high=fs / 2)
        self.n_components = n_components
        self.relative = relative

    def apply(self, x, generator=None):
        b, c, _ = x.shape
        t = time_axis(x, self.fs)
        wander = torch.zeros_like(x)
        for _ in range(self.n_components):
            freq = uniform(self.freq, (b, 1, 1), x, generator)
            phase = uniform((0, 2 * math.pi), (b, c, 1), x, generator)
            amp = uniform(self.amplitude, (b, c, 1), x, generator) / self.n_components
            wander = wander + amp * torch.sin(2 * math.pi * freq * t + phase)
        if self.relative:
            wander = wander * x.std(dim=-1, keepdim=True, unbiased=False)
        return x + wander


class PowerlineNoise(Transform):
    """Add mains interference at a fundamental frequency and its harmonics.

    Args:
        fs: sampling rate (Hz).
        freq: mains frequency or (low, high) range in Hz, e.g. 50 or 60.
        snr_db: signal-to-noise ratio or (low, high) range in dB, per lead.
        harmonics: number of harmonics including the fundamental (amplitudes 1/k).
    """

    def __init__(self, fs, freq=50.0, snr_db=(20.0, 40.0), harmonics=1, p=1.0):
        super().__init__(p)
        self.fs = fs
        self.freq = as_range(freq, "freq", low=0, high=fs / 2 / harmonics - 1e-6)
        self.snr_db = as_range(snr_db, "snr_db")
        self.harmonics = harmonics

    def apply(self, x, generator=None):
        b, c, _ = x.shape
        t = time_axis(x, self.fs)
        freq = uniform(self.freq, (b, 1, 1), x, generator)
        noise = torch.zeros_like(x)
        for k in range(1, self.harmonics + 1):
            phase = uniform((0, 2 * math.pi), (b, c, 1), x, generator)
            noise = noise + torch.sin(2 * math.pi * k * freq * t + phase) / k
        snr = uniform(self.snr_db, (b, 1, 1), x, generator)
        return add_noise(x, noise, snr)
