import numpy as np
import torch

from .base import Transform, randint


class Compose(Transform):
    """Apply transforms in sequence.

    Args:
        transforms: list of transforms, or a single transform.
    """

    def __init__(self, transforms, p=1.0):
        super().__init__(p)
        if isinstance(transforms, torch.nn.Module):
            transforms = [transforms]
        self.transforms = torch.nn.ModuleList(transforms)

    def apply(self, x, generator=None):
        for t in self.transforms:
            x = t(x, generator=generator)
        return x

    def extra_repr(self):
        return f"p={self.p}"


class RandomApply(Compose):
    """Apply transforms in sequence with probability p, decided per sample."""

    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p)


class OneOf(Compose):
    """Apply one transform chosen uniformly at random per sample."""

    def apply(self, x, generator=None):
        choice = randint(0, len(self.transforms), x.shape[0], x, generator)
        out = x.clone()
        for k, t in enumerate(self.transforms):
            chosen = choice == k
            if chosen.any():
                out[chosen] = t(x[chosen], generator=generator)
        return out


class ComposeMany(Compose):
    """Apply transforms independently N times, stacking the views along a new first axis.

    Args:
        transforms: list of transforms, or a single transform.
        num_augmented_samples: number of views N.
    """

    def __init__(self, transforms, num_augmented_samples):
        super().__init__(transforms)
        self.num_augmented_samples = num_augmented_samples

    def forward(self, x, generator=None):
        forward = super().forward
        views = [forward(x, generator) for _ in range(self.num_augmented_samples)]
        return np.stack(views) if isinstance(x, np.ndarray) else torch.stack(views)

    def extra_repr(self):
        return f"num_augmented_samples={self.num_augmented_samples}"
