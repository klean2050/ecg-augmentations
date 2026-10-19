import numpy as np
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
    rate_exponent = 0.0

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

    def windows(self, peaks):
        """Per-beat (start, end) samples; the part after R scales with RR^rate_exponent."""
        peaks = np.asarray(peaks, dtype=int)
        rr = np.full(len(peaks), self.fs)
        if len(peaks) > 1:
            rr = np.diff(peaks, prepend=2 * peaks[0] - peaks[1])
        after = self.window[1] * (rr / self.fs) ** self.rate_exponent
        before = round(self.window[0] * self.fs)
        return peaks - before, peaks + np.round(after * self.fs).astype(int)

    def make_mask(self, x, generator=None):
        b, _, t = x.shape
        mask = torch.zeros(b, 1, t, dtype=torch.bool, device=x.device)
        for i in range(b):
            peaks = self.detector(x[i, self.lead].detach().cpu().numpy(), self.fs)
            chosen = uniform((0, 1), len(peaks), x, generator) < self.ratio
            for start, end, keep in zip(*self.windows(peaks), chosen.tolist()):
                if keep:
                    mask[i, 0, max(start, 0) : min(end, t)] = True
        return mask


class PRMask(BeatMask):
    """Mask 0.2 s before each R peak (P wave and PR segment)."""

    window = (0.2, 0.0)


class QRSMask(BeatMask):
    """Mask 0.05 s around each R peak (QRS complex)."""

    window = (0.05, 0.05)


class QTMask(BeatMask):
    """Mask the QRS, ST segment and T wave: from 0.05 s before each R peak to 0.33 s
    after it at 60 bpm, scaled with the preceding RR interval as sqrt(RR) (Bazett)."""

    window = (0.05, 0.33)
    rate_exponent = 0.5
