import torch

from ..base import Transform, as_range, uniform
from ..ops import interp


class TimeStretch(Transform):
    """Stretch the signal in time by a random factor, keeping its length.

    A factor k > 1 lengthens all intervals (lower heart rate) and drops the end;
    k < 1 shortens them and pads the end by repeating the last sample.

    Args:
        factor: stretch factor or (low, high) range, all positive.
    """

    def __init__(self, factor=(0.9, 1.1), p=1.0):
        super().__init__(p)
        self.factor = as_range(factor, "factor", low=1e-6)

    def apply(self, x, generator=None):
        factor = uniform(self.factor, (x.shape[0], 1, 1), x, generator)
        return interp(
            x, torch.arange(x.shape[-1], device=x.device, dtype=x.dtype) / factor
        )
