import math, torch
import numpy as np

from ..base import Transform, as_range, randint, uniform
from .mask import MaskTransform

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT = (0, 1, 6, 7, 8, 9, 10, 11)

# Kors regression matrix: I, II, V1-V6 -> X, Y, Z vectorcardiogram
KORS = (
    (0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54),
    (-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13),
    (0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31),
)

# Reduced lead sets of the PhysioNet/CinC Challenge 2021, plus lead I alone
LEAD_SETS = {
    "12": LEADS,
    "6": ("I", "II", "III", "aVR", "aVL", "aVF"),
    "4": ("I", "II", "III", "V2"),
    "3": ("I", "II", "V2"),
    "2": ("I", "II"),
    "1": ("I",),
}


def check_12_leads(x):
    if x.shape[1] != 12:
        raise ValueError(f"expected 12 leads in the order {LEADS}, got {x.shape[1]}")


def derive_limb_leads(x):
    """Recompute III, aVR, aVL and aVF from leads I and II of a (B, 12, T) signal."""
    i, ii = x[:, 0], x[:, 1]
    derived = torch.stack([ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2], dim=1)
    return torch.cat([x[:, :2], derived, x[:, 6:]], dim=1)


def rotation(angles):
    """Rotation matrices (B, 3, 3) from angles (B, 3) in radians about X, Y and Z."""
    cos, sin = angles.cos(), angles.sin()
    one, zero = torch.ones_like(cos[:, 0]), torch.zeros_like(cos[:, 0])

    def matrix(rows):
        return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)

    (cx, cy, cz), (sx, sy, sz) = cos.unbind(-1), sin.unbind(-1)
    rx = matrix([(one, zero, zero), (zero, cx, -sx), (zero, sx, cx)])
    ry = matrix([(cy, zero, sy), (zero, one, zero), (-sy, zero, cy)])
    rz = matrix([(cz, -sz, zero), (sz, cz, zero), (zero, zero, one)])
    return rz @ ry @ rx


class DeriveLimbLeads(Transform):
    """Make III, aVR, aVL and aVF consistent with leads I and II (12-lead input).

    Useful after per-lead transforms, which break Einthoven and Goldberger relations.
    """

    def apply(self, x, generator=None):
        check_12_leads(x)
        return derive_limb_leads(x)


class VCGRotation(Transform):
    """Rotate and scale the heart's electrical axis in vectorcardiogram space.

    Leads I, II and V1-V6 are mapped to X, Y, Z with the Kors matrix, rotated and
    scaled there, and mapped back; the part of the signal outside the VCG subspace
    is kept, and the other limb leads are rederived. Input: 12 leads in the
    standard order. Inspired by 3KG (Gopal et al., 2021).

    Args:
        angle: maximum rotation in degrees about each axis, or (low, high) range.
        scale: scale factor or (low, high) range, drawn per axis.
    """

    def __init__(self, angle=15.0, scale=(0.9, 1.1), p=1.0):
        super().__init__(p)
        self.angle = as_range((-angle, angle) if np.isscalar(angle) else angle, "angle")
        self.scale = as_range(scale, "scale", low=1e-6)

    def apply(self, x, generator=None):
        check_12_leads(x)
        b = x.shape[0]
        angles = uniform(self.angle, (b, 3), x, generator) * math.pi / 180
        scales = uniform(self.scale, (b, 3), x, generator)
        kors = torch.tensor(KORS, dtype=x.dtype, device=x.device)
        transform = rotation(angles) * scales[:, None, :] - torch.eye(3).to(x)
        change = torch.linalg.pinv(kors) @ transform @ kors
        independent = x[:, INDEPENDENT]
        out = x.clone()
        out[:, INDEPENDENT] = independent + change @ independent
        return derive_limb_leads(out)


class LeadSubset(MaskTransform):
    """Keep a random reduced lead set and mask the other leads (12-lead input).

    Args:
        subsets: names of sets in LEAD_SETS to choose from uniformly, e.g. ("6", "1").
        fill: "zero" or "mean" (lead mean).
        return_mask: also return the boolean mask of masked samples.
    """

    def __init__(
        self, subsets=("6", "4", "3", "2", "1"), fill="zero", return_mask=False, p=1.0
    ):
        super().__init__(fill, return_mask, p)
        unknown = set(subsets) - set(LEAD_SETS)
        if unknown or not subsets:
            raise ValueError(
                f"unknown lead sets {unknown}; choose from {list(LEAD_SETS)}"
            )
        self.subsets = tuple(subsets)

    def make_mask(self, x, generator=None):
        check_12_leads(x)
        keep = torch.tensor(
            [[lead in LEAD_SETS[s] for lead in LEADS] for s in self.subsets]
        )
        choice = randint(0, len(self.subsets), x.shape[0], x, generator)
        return ~keep.to(x.device)[choice][..., None]
