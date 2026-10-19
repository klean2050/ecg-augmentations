import math

from ..base import Transform, as_range, uniform


class AmplitudeScale(Transform):
    """Multiply the signal by a log-uniform random factor.

    Args:
        scale: factor or (low, high) range, all positive.
        per_lead: draw an independent factor for each lead.
    """

    def __init__(self, scale=(0.5, 2.0), per_lead=False, p=1.0):
        super().__init__(p)
        self.scale = as_range(scale, "scale", low=1e-12)
        self.per_lead = per_lead

    def apply(self, x, generator=None):
        shape = (x.shape[0], x.shape[1] if self.per_lead else 1, 1)
        log_scale = uniform(tuple(map(math.log, self.scale)), shape, x, generator)
        return x * log_scale.exp()
