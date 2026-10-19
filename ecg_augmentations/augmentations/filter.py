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
