import torch


def add_noise(x, noise, snr_db):
    """Add zero-mean noise scaled to a per-lead SNR (dB) relative to the signal power."""
    signal = x.var(dim=-1, keepdim=True, unbiased=False)
    noise = noise - noise.mean(dim=-1, keepdim=True)
    power = noise.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    return x + torch.sqrt(signal / power / 10 ** (snr_db / 10)) * noise


def fft_filter(x, fs, gain, pad=True):
    """Zero-phase filter along time; gain maps frequencies (Hz) to amplitude gain.

    With pad, the signal is extended by odd reflection to limit edge effects;
    without, filtering is circular, which suits generated stationary noise.
    """
    n = x.shape[-1]
    p = n - 1 if pad else 0
    left = 2 * x[..., :1] - x[..., 1 : p + 1].flip(-1)
    right = 2 * x[..., -1:] - x[..., n - p - 1 : n - 1].flip(-1)
    xp = torch.cat([left, x, right], dim=-1)
    freqs = torch.fft.rfftfreq(xp.shape[-1], d=1 / fs).to(x.device, x.dtype)
    y = torch.fft.irfft(torch.fft.rfft(xp) * gain(freqs), n=xp.shape[-1])
    return y[..., p : p + n]


def butter_gain(cutoff, order, highpass=False):
    """Butterworth magnitude response with cutoff (Hz) of any broadcastable shape."""

    def gain(freqs):
        ratio = cutoff / freqs if highpass else freqs / cutoff
        return 1 / torch.sqrt(1 + ratio ** (2 * order))

    return gain


def interp(x, pos):
    """Linearly interpolate x (B, C, T) at fractional sample positions pos (B, 1, N)."""
    n = x.shape[-1]
    pos = pos.clamp(0, n - 1)
    i0 = pos.floor().long().clamp(max=max(n - 2, 0))
    i1 = (i0 + 1).clamp(max=n - 1)
    w = pos - i0
    shape = (-1, x.shape[1], -1)
    return x.gather(-1, i0.expand(shape)) * (1 - w) + x.gather(-1, i1.expand(shape)) * w


def fill(x, mask, mode="zero"):
    """Replace masked samples by zero, the lead mean, or the last unmasked value."""
    mask = mask.expand_as(x)
    if mode == "zero":
        value = torch.zeros_like(x)
    elif mode == "mean":
        value = x.mean(dim=-1, keepdim=True).expand_as(x)
    elif mode == "hold":
        idx = torch.arange(x.shape[-1], device=x.device).expand_as(x)
        last = torch.where(mask, -1, idx).cummax(dim=-1).values.clamp(min=0)
        value = x.gather(-1, last)
    else:
        raise ValueError(f"unknown fill mode: {mode}")
    return torch.where(mask, value, x)
