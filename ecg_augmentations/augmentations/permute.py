import torch

from ..base import Transform, randperm


class Permute(Transform):
    """Split the signal into random segments and shuffle them, shared across leads.

    Args:
        n_patches: number of segments.
    """

    def __init__(self, n_patches=10, p=1.0):
        super().__init__(p)
        if n_patches < 1:
            raise ValueError(f"n_patches must be positive, got {n_patches}")
        self.n_patches = n_patches

    def apply(self, x, generator=None):
        t = x.shape[-1]
        if self.n_patches > t:
            raise ValueError(f"cannot split {t} samples into {self.n_patches} patches")
        out = []
        for sample in x:
            cuts = randperm(t - 1, generator)[: self.n_patches - 1] + 1
            sizes = torch.diff(
                torch.cat([torch.tensor([0]), cuts.sort().values, torch.tensor([t])])
            )
            patches = sample.split(sizes.tolist(), dim=-1)
            order = randperm(self.n_patches, generator)
            out.append(torch.cat([patches[i] for i in order], dim=-1))
        return torch.stack(out)
