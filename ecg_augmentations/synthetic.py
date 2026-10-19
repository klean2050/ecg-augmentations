import torch

from .augmentations.leads import derive_limb_leads

# P, Q, R, S, T waves: (offset from R peak in s, amplitude, width in s)
WAVES = (
    (-0.2, 0.15, 0.025),
    (-0.03, -0.15, 0.01),
    (0.0, 1.0, 0.012),
    (0.03, -0.25, 0.01),
    (0.3, 0.3, 0.05),
)


def synthetic_ecg(
    duration=10.0, fs=500, heart_rate=60.0, n_leads=12, hrv=0.02, seed=None
):
    """Simulate a multi-lead ECG as a sum of Gaussian P-QRS-T waves.

    Args:
        duration: length in seconds.
        fs: sampling rate (Hz).
        heart_rate: mean heart rate in beats per minute.
        n_leads: number of leads; lead 0 has unit gains, others random positive gains.
            With 12 leads, III, aVR, aVL and aVF are derived from I and II.
        hrv: standard deviation of RR intervals in seconds.
        seed: seed for the RR intervals and lead gains.

    Returns:
        signal (n_leads, round(duration * fs)) float32 tensor in mV-like units,
        and R-peak sample indices as a long tensor.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    rr = 60.0 / heart_rate
    n_beats = int(duration / rr) + 2
    peaks = 0.4 + torch.cumsum(rr + hrv * torch.randn(n_beats, generator=g), 0) - rr
    peaks = peaks[peaks < duration]

    t = torch.arange(round(duration * fs)) / fs
    offset, amp, width = (torch.tensor(v)[:, None, None] for v in zip(*WAVES))
    delay = t - peaks[:, None] - offset
    waves = (amp * torch.exp(-(delay**2) / (2 * width**2))).sum(1)
    gains = 0.4 + 0.8 * torch.rand(n_leads, len(WAVES), generator=g)
    gains[0] = 1.0
    signal = (gains @ waves).float()
    if n_leads == 12:
        signal = derive_limb_leads(signal[None])[0]
    return signal, (peaks * fs).round().long()
