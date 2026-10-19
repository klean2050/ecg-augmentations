import torch

from ..base import Transform, as_range, randint, to_batched, uniform
from ..ops import fill


def neurokit_peaks(signal, fs):
    """Detect R peaks with neurokit2 (install with `pip install ecg-augmentations[peaks]`)."""
    try:
        import neurokit2 as nk
    except ImportError as e:
        raise ImportError(
            "R-peak detection needs neurokit2: pip install ecg-augmentations[peaks]"
        ) from e
    _, info = nk.ecg_peaks(signal, sampling_rate=fs)
    return info["ECG_R_Peaks"]


class MaskTransform(Transform):
    """Base for masks; subclasses implement `make_mask(x, generator)` -> bool (B, 1 or C, T or 1)."""

    def __init__(self, fill="zero", return_mask=False, p=1.0):
        super().__init__(p)
        self.fill = fill
        self.return_mask = return_mask

    def apply(self, x, generator=None):
        return fill(x, self.make_mask(x, generator), self.fill)

    def forward(self, x, generator=None):
        if not self.return_mask:
            return super().forward(x, generator)
        x, restore = to_batched(x)
        mask = self.make_mask(x, generator) & self.gate(x, generator)[:, None, None]
        return restore(fill(x, mask, self.fill)), restore(mask.expand_as(x))


class TimeMask(MaskTransform):
    """Mask random time windows, shared across leads.

    Args:
        fs: sampling rate (Hz).
        duration: window length or (low, high) range in seconds.
        n_masks: number of windows or (low, high) range.
        fill: "zero", "mean" (lead mean) or "hold" (last unmasked value).
        return_mask: also return the boolean mask of masked samples.
    """

    def __init__(
        self, fs, duration=(0.1, 0.5), n_masks=1, fill="zero", return_mask=False, p=1.0
    ):
        super().__init__(fill, return_mask, p)
        self.fs = fs
        self.duration = as_range(duration, "duration", low=0)
        self.n_masks = tuple(int(n) for n in as_range(n_masks, "n_masks", low=0))

    def make_mask(self, x, generator=None):
        b, _, t = x.shape
        idx = torch.arange(t, device=x.device)
        count = randint(self.n_masks[0], self.n_masks[1] + 1, (b, 1, 1), x, generator)
        mask = torch.zeros(b, 1, t, dtype=torch.bool, device=x.device)
        for k in range(self.n_masks[1]):
            seconds = uniform(self.duration, (b, 1, 1), x, generator)
            length = (seconds * self.fs).round().clamp(max=t)
            start = randint(0, t - length + 1, (b, 1, 1), x, generator)
            mask |= (idx >= start) & (idx < start + length) & (k < count)
        return mask


class LeadMask(MaskTransform):
    """Mask entire random leads; at least one lead is always kept.

    Args:
        n_leads: number of masked leads or (low, high) range.
        fill: "zero" or "mean" (lead mean).
        return_mask: also return the boolean mask of masked samples.
    """

    def __init__(self, n_leads=1, fill="zero", return_mask=False, p=1.0):
        super().__init__(fill, return_mask, p)
        self.n_leads = tuple(int(n) for n in as_range(n_leads, "n_leads", low=0))

    def make_mask(self, x, generator=None):
        b, c, _ = x.shape
        low, high = (min(n, c - 1) for n in self.n_leads)
        count = randint(low, high + 1, (b, 1), x, generator)
        rank = uniform((0, 1), (b, c), x, generator).argsort(dim=-1).argsort(dim=-1)
        return (rank < count)[..., None]


class BeatMask(MaskTransform):
    """Mask a fixed window around detected R peaks, shared across leads.

    Args:
        fs: sampling rate (Hz).
        ratio: probability that each beat is masked.
        detector: callable (signal_1d_numpy, fs) -> R-peak indices; defaults to neurokit2.
        lead: lead used for R-peak detection.
        fill: "zero", "mean" (lead mean) or "hold" (last unmasked value).
        return_mask: also return the boolean mask of masked samples.
    """

    window = (0.0, 0.0)

    def __init__(
        self,
        fs,
        ratio=0.5,
        detector=None,
        lead=0,
        fill="hold",
        return_mask=False,
        p=1.0,
    ):
        super().__init__(fill, return_mask, p)
        if not 0 <= ratio <= 1:
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")
        self.fs = fs
        self.ratio = ratio
        self.detector = detector or neurokit_peaks
        self.lead = lead

    def make_mask(self, x, generator=None):
        b, _, t = x.shape
        before, after = (round(w * self.fs) for w in self.window)
        mask = torch.zeros(b, 1, t, dtype=torch.bool, device=x.device)
        for i in range(b):
            peaks = self.detector(x[i, self.lead].detach().cpu().numpy(), self.fs)
            chosen = uniform((0, 1), len(peaks), x, generator) < self.ratio
            for r, keep in zip(peaks, chosen.tolist()):
                if keep:
                    mask[i, 0, max(int(r) - before, 0) : min(int(r) + after, t)] = True
        return mask


class PRMask(BeatMask):
    """Mask 0.2 s before each R peak (P wave and PR segment)."""

    window = (0.2, 0.0)


class QRSMask(BeatMask):
    """Mask 0.05 s around each R peak (QRS complex)."""

    window = (0.05, 0.05)


class QTMask(BeatMask):
    """Mask from 0.05 s before to 0.33 s after each R peak (QRS, ST segment and T wave)."""

    window = (0.05, 0.33)
