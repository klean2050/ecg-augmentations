from ..base import Transform


class Invert(Transform):
    """Flip the signal polarity."""

    def apply(self, x, generator=None):
        return -x
