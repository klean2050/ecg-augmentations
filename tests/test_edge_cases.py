import pytest, torch

import ecg_augmentations as ea


@pytest.mark.parametrize(
    "make",
    [
        lambda: ea.Invert(p=1.5),
        lambda: ea.GaussianNoise(snr_db=(30, 10)),
        lambda: ea.HighPassFilter(100, cutoff=60),
        lambda: ea.LowPassFilter(100, cutoff=(20, 50)),
        lambda: ea.PowerlineNoise(100, freq=50),
        lambda: ea.AmplitudeScale((0, 2)),
        lambda: ea.TimeStretch(0),
        lambda: ea.RandomCrop(),
        lambda: ea.RandomCrop(10, duration=1, fs=100),
        lambda: ea.RandomCrop(duration=1),
        lambda: ea.Permute(0),
        lambda: ea.PRMask(100, ratio=2),
        lambda: ea.FrequencyMask(100, width=60),
        lambda: ea.RandomResizedCrop(100, scale=(0.5, 1.5)),
        lambda: ea.Mixup(alpha=0),
    ],
)
def test_invalid_parameters(make):
    with pytest.raises(ValueError):
        make()


def test_invalid_call_sizes():
    with pytest.raises(ValueError):
        ea.RandomCrop(200)(torch.zeros(1, 100))
    with pytest.raises(ValueError):
        ea.Permute(20)(torch.zeros(1, 10))


def test_unknown_fill():
    with pytest.raises(ValueError):
        ea.TimeMask(100, fill="nan")(torch.zeros(1, 100))


@pytest.mark.parametrize("length", [1, 2, 5])
def test_short_signals(transform, length):
    if isinstance(transform, (ea.RandomCrop, ea.RandomResizedCrop, ea.Permute)):
        pytest.skip("length-constrained")
    y = transform(torch.ones(12, length))
    assert y.shape == (12, length) and torch.isfinite(y).all()


def test_constant_signal_no_nan(transform):
    assert torch.isfinite(transform(torch.full((12, 2500), 3.0))).all()


def test_mask_longer_than_signal():
    y = ea.TimeMask(100, duration=5.0)(torch.ones(2, 50))
    assert torch.equal(y, torch.zeros(2, 50))


def test_missing_neurokit_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "neurokit2":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="ecg-augmentations\\[peaks\\]"):
        ea.QRSMask(100)(torch.zeros(1, 1000))
