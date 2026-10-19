import torch

from ..base import Transform


class Reverse(Transform):
    """Reverse the signal in time."""

    def apply(self, x, generator=None):
        return torch.flip(x, dims=[-1])
