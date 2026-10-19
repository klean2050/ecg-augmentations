import torch
import torch.nn.functional as F

from ..base import Transform, as_range, randn, uniform
from ..ops import interp


class TimeStretch(Transform):
    """Stretch the signal in time by a random factor, keeping its length.

    A factor k > 1 lengthens all intervals (lower heart rate) and keeps a random
    window of the stretched signal; k < 1 shortens them and pads both ends, by a
    random split, with the edge values.

    Args:
        factor: stretch factor or (low, high) range, all positive.
    """

    def __init__(self, factor=(0.9, 1.1), p=1.0):
        super().__init__(p)
        self.factor = as_range(factor, "factor", low=1e-6)

    def apply(self, x, generator=None):
        b, _, t = x.shape
        factor = uniform(self.factor, (b, 1, 1), x, generator)
        offset = uniform((0, 1), (b, 1, 1), x, generator) * (t - 1) * (1 - 1 / factor)
        idx = torch.arange(t, device=x.device, dtype=x.dtype)
        return interp(x, offset + idx / factor)


class RandomTimeWarp(Transform):
    """Warp time smoothly and monotonically, shared across leads; endpoints are kept.

    Args:
        sigma: strength or (low, high) range; local speed varies roughly by exp(±sigma).
        n_knots: number of random control points of the warp.
    """

    def __init__(self, sigma=(0.0, 0.3), n_knots=4, p=1.0):
        super().__init__(p)
        self.sigma = as_range(sigma, "sigma", low=0)
        self.n_knots = n_knots

    def apply(self, x, generator=None):
        b, _, t = x.shape
        sigma = uniform(self.sigma, (b, 1, 1), x, generator)
        knots = randn((b, 1, self.n_knots), x, generator)
        speed = torch.exp(
            sigma * F.interpolate(knots, t, mode="linear", align_corners=True)
        )
        pos = torch.cumsum(speed, dim=-1) - speed[..., :1]
        return interp(x, pos / pos[..., -1:].clamp_min(1e-12) * (t - 1))


class TimeShift(Transform):
    """Shift the signal in time, shared across leads.

    Args:
        fs: sampling rate (Hz).
        shift: shift or (low, high) range in seconds; positive delays the signal.
        circular: wrap samples around; otherwise vacated samples are zero.
    """

    def __init__(self, fs, shift=(-0.5, 0.5), circular=True, p=1.0):
        super().__init__(p)
        self.fs = fs
        self.shift = as_range(shift, "shift")
        self.circular = circular

    def apply(self, x, generator=None):
        b, c, t = x.shape
        shift = (uniform(self.shift, (b, 1, 1), x, generator) * self.fs).round().long()
        idx = torch.arange(t, device=x.device) - shift
        out = x.gather(-1, (idx % t).expand(b, c, t))
        return out if self.circular else out * ((idx >= 0) & (idx < t))
