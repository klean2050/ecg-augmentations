import numpy as np
import torch


def to_batched(x):
    """Return x as a (B, C, T) tensor and a function restoring the input format."""
    is_numpy = isinstance(x, np.ndarray)
    x = torch.as_tensor(x)
    if not x.is_floating_point():
        raise TypeError(f"expected a floating-point signal, got {x.dtype}")
    if x.ndim not in (1, 2, 3):
        raise ValueError(
            f"expected shape (T,), (C, T) or (B, C, T), got {tuple(x.shape)}"
        )
    shape = x.shape

    def restore(y):
        y = y.reshape(shape[:-1] + y.shape[-1:])
        return y.detach().cpu().numpy() if is_numpy else y

    return x.reshape((1,) * (3 - x.ndim) + tuple(shape)), restore


def as_range(value, name, low=None, high=None):
    """Return a scalar or (low, high) pair as a validated float pair."""
    lo, hi = (value, value) if np.isscalar(value) else value
    if lo > hi or (low is not None and lo < low) or (high is not None and hi > high):
        raise ValueError(f"invalid {name}: {value}")
    return float(lo), float(hi)


def sample(fn, shape, like, generator=None):
    """Draw from a torch sampler on the generator's device, then move to `like`."""
    device = generator.device if generator is not None else like.device
    out = fn(shape, generator=generator, device=device, dtype=like.dtype)
    return out.to(like.device)


def uniform(bounds, shape, like, generator=None):
    """Sample uniformly in bounds, on the device and dtype of `like`."""
    return bounds[0] + (bounds[1] - bounds[0]) * sample(
        torch.rand, shape, like, generator
    )


def randn(shape, like, generator=None):
    """Sample standard normal noise, on the device and dtype of `like`."""
    return sample(torch.randn, shape, like, generator)


def randint(low, high, shape, like, generator=None):
    """Sample integers in [low, high), with high a scalar or a tensor."""
    ref = torch.empty(0, dtype=torch.float64, device=like.device)
    return (low + uniform((0, 1), shape, ref, generator) * (high - low)).long()


def randperm(n, generator=None):
    device = generator.device if generator is not None else "cpu"
    return torch.randperm(n, generator=generator, device=device).cpu()


class Transform(torch.nn.Module):
    """Base transform: applied to each sample with probability p.

    Subclasses implement `apply(x, generator)` for x of shape (B, C, T).
    """

    def __init__(self, p=1.0):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p

    def gate(self, x, generator=None):
        return uniform((0, 1), x.shape[0], x, generator) < self.p

    def forward(self, x, generator=None):
        x, restore = to_batched(x)
        chosen = self.gate(x, generator)
        if chosen.all():
            return restore(self.apply(x, generator))
        out = x.clone()
        if chosen.any():
            out[chosen] = self.apply(x[chosen], generator)
        return restore(out)

    def apply(self, x, generator=None):
        raise NotImplementedError

    def extra_repr(self):
        return ", ".join(
            f"{k}={v}"
            for k, v in vars(self).items()
            if not k.startswith("_") and k != "training"
        )
