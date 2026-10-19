import math
import pytest, torch

import ecg_augmentations as ea
from tests.common import FS, PEAKS, detector


def snr_db(x, y):
    noise = y - x
    return 10 * torch.log10(x.var(-1, unbiased=False) / noise.var(-1, unbiased=False))


def spectrum(x):
    window = torch.hann_window(x.shape[-1], dtype=x.dtype)
    return (
        torch.fft.rfftfreq(x.shape[-1], 1 / FS),
        torch.fft.rfft(x * window).abs() ** 2,
    )


def sine(freq, n=10 * FS):
    return torch.sin(2 * math.pi * freq * torch.arange(n, dtype=torch.float64) / FS)


@pytest.mark.parametrize("target", [5.0, 20.0, 40.0])
def test_gaussian_snr(ecg, target):
    assert torch.allclose(
        snr_db(ecg, ea.GaussianNoise(target)(ecg)), torch.tensor(target), atol=0.01
    )


def test_powerline_frequency(ecg):
    freqs, power = spectrum(ea.PowerlineNoise(FS, freq=60, snr_db=0)(ecg) - ecg)
    assert torch.all(freqs[power.argmax(-1)] == 60)
    assert torch.allclose(
        snr_db(ecg, ea.PowerlineNoise(FS, snr_db=10)(ecg)),
        torch.tensor(10.0),
        atol=0.01,
    )


def test_constant_lead_gets_no_noise():
    x = torch.zeros(2, 500)
    assert torch.equal(ea.GaussianNoise(10)(x), x)
    assert torch.equal(ea.BaselineWander(FS)(x), x)


def test_wander_is_low_frequency(ecg):
    freqs, power = spectrum(
        ea.BaselineWander(FS, amplitude=1.0, freq=(0.1, 0.5))(ecg) - ecg
    )
    assert power[..., freqs > 1].sum() < 1e-3 * power.sum()
    wander = ea.BaselineWander(
        FS, amplitude=0.3, freq=0.5, n_components=1, relative=False
    )(torch.zeros(3, 2500))
    assert torch.allclose(wander.abs().amax(-1), torch.tensor(0.3), atol=1e-3)


def test_highpass():
    x = sine(0.1) + sine(10)
    y = ea.HighPassFilter(FS, cutoff=1.0)(x)
    assert torch.allclose(y[500:-500], sine(10)[500:-500], atol=0.02)


def test_lowpass():
    x = sine(5) + sine(100)
    y = ea.LowPassFilter(FS, cutoff=25.0)(x)
    assert torch.allclose(y[100:-100], sine(5)[100:-100], atol=0.01)
    freqs, power = spectrum(y)
    assert power[freqs == 100] < 0.01 * power[freqs == 5]


def test_amplitude_scale(ecg):
    y = ea.AmplitudeScale((0.5, 2.0))(ecg)
    ratio = y / ecg
    assert torch.allclose(ratio[ecg.abs() > 0.1], ratio[0, PEAKS[0]], rtol=1e-5)
    assert 0.5 <= ratio[0, PEAKS[0]] <= 2.0


def test_involutions(ecg):
    assert torch.equal(ea.Invert()(ea.Invert()(ecg)), ecg)
    assert torch.equal(ea.Reverse()(ea.Reverse()(ecg)), ecg)
    assert torch.equal(ea.Reverse()(ecg)[:, 0], ecg[:, -1])


def test_permute_preserves_values(ecg):
    y = ea.Permute(n_patches=7)(ecg)
    assert torch.equal(y.sort(-1).values, ecg.sort(-1).values)
    assert not torch.equal(y, ecg)
    assert torch.equal(
        (y[0:1] == ecg[0:1, :, None]).any(1),
        torch.ones(1, ecg.shape[-1], dtype=torch.bool),
    )


def test_permute_shared_across_leads():
    x = torch.arange(100.0).expand(3, -1)
    y = ea.Permute(n_patches=5)(x)
    assert torch.equal(y[0], y[1]) and torch.equal(y[1], y[2])


def test_random_crop_is_window(ecg):
    y = ea.RandomCrop(300)(ecg)
    start = int((ecg[0] == y[0, 0]).nonzero()[0])
    assert torch.equal(y, ecg[:, start : start + 300])


def test_time_stretch(ecg):
    assert torch.allclose(ea.TimeStretch(1.0)(ecg), ecg, atol=1e-6)
    y = ea.TimeStretch(1.25)(ecg)
    stretched = (PEAKS * 1.25).round().long()
    stretched = stretched[stretched < ecg.shape[-1]]
    assert torch.allclose(y[0, stretched], ecg[0, PEAKS[: len(stretched)]], atol=0.15)
    assert torch.all(y[0, stretched] > 0.8)


def test_time_mask_fraction(ecg):
    _, mask = ea.TimeMask(FS, duration=0.4, n_masks=2, return_mask=True)(ecg)
    assert 0.4 * FS <= mask[0].sum() <= 0.8 * FS
    assert torch.equal(mask[0], mask[5])


def test_lead_mask(ecg):
    y = ea.LeadMask(n_leads=4)(ecg)
    masked = (y == 0).all(-1)
    assert masked.sum() == 4
    assert torch.equal(y[~masked], ecg[~masked])
    single = ecg[:1]
    assert torch.equal(ea.LeadMask(n_leads=1)(single), single)


@pytest.mark.parametrize(
    "cls,window",
    [(ea.PRMask, (0.2, 0)), (ea.QRSMask, (0.05, 0.05)), (ea.QTMask, (0.05, 0.33))],
)
def test_beat_masks(ecg, cls, window):
    y = cls(FS, ratio=1.0, detector=detector)(ecg)
    allowed = torch.zeros(ecg.shape[-1], dtype=torch.bool)
    for r in PEAKS.tolist():
        allowed[max(r - round(window[0] * FS), 0) : r + round(window[1] * FS)] = True
    changed = (y != ecg).any(0)
    assert changed.any() and not changed[~allowed].any()
    assert torch.equal(cls(FS, ratio=0.0, detector=detector)(ecg), ecg)


def test_hold_fill():
    x = torch.arange(10.0)[None]
    y = ea.QRSMask(20, ratio=1.0, detector=lambda s, fs: [5], fill="hold")(x)
    assert y[0].tolist() == [0, 1, 2, 3, 3, 3, 6, 7, 8, 9]


def test_beat_mask_near_start_does_not_wrap():
    x = torch.ones(1, 100)
    y = ea.PRMask(100, ratio=1.0, detector=lambda s, fs: [5], fill="zero")(x)
    assert y[0, :5].eq(0).all() and y[0, 5:].eq(1).all()


def test_muscle_noise(ecg):
    y = ea.MuscleNoise(FS, snr_db=15, bursts=False)(ecg)
    assert torch.allclose(snr_db(ecg, y), torch.tensor(15.0), atol=0.01)
    freqs, power = spectrum(y - ecg)
    assert power[..., freqs < 15].sum() < 0.01 * power.sum()
    assert torch.isfinite(ea.MuscleNoise(FS, bursts=True)(ecg)).all()


def test_baseline_shift_is_step(ecg):
    step = ea.BaselineShift(amplitude=0.5, relative=False)(ecg) - ecg
    jumps = (step.diff(dim=-1).abs() > 1e-6).sum(-1)
    assert torch.all(jumps <= 1)
    assert torch.allclose(step[:, -1].abs(), torch.tensor(0.5), atol=1e-6)
    assert torch.equal(jumps, jumps[0].expand_as(jumps))


def test_recorded_noise(ecg):
    bank = torch.randn(2, 5000, dtype=torch.float64)
    assert torch.allclose(
        snr_db(ecg, ea.RecordedNoise(bank, snr_db=12)(ecg)),
        torch.tensor(12.0),
        atol=0.01,
    )
    with pytest.raises(ValueError):
        ea.RecordedNoise(bank[:, :100])(ecg)
    sines = sine(7)[None].float()
    freqs, power = spectrum(
        ea.RecordedNoise(sines, snr_db=0)(ecg[:, :1000]) - ecg[:, :1000]
    )
    assert torch.all(freqs[power.argmax(-1)].round() == 7)


def test_random_time_warp(ecg):
    x = ecg.double()
    assert torch.allclose(ea.RandomTimeWarp(sigma=0)(x), x, atol=1e-8)
    y = ea.RandomTimeWarp(sigma=0.5)(ecg)
    assert not torch.allclose(y, ecg, atol=1e-3)
    assert torch.allclose(y[:, [0, -1]], ecg[:, [0, -1]], atol=1e-5)


def test_time_shift(ecg):
    assert torch.equal(ea.TimeShift(FS, shift=0.2)(ecg), ecg.roll(50, dims=-1))
    y = ea.TimeShift(FS, shift=-0.2, circular=False)(ecg)
    assert torch.equal(y[:, :-50], ecg[:, 50:]) and y[:, -50:].eq(0).all()
