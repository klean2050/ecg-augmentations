import pytest, torch
import neurokit2 as nk
from ecg_augmentations import *


dur, sr = 10, 100
num_samples = dur * sr


@pytest.mark.parametrize("num_channels", [1, 2])
def test_compose(num_channels):
    ecg = nk.ecg_simulate(duration=dur, sampling_rate=sr)
    transform = Compose(
        [
            RandomCrop(num_samples),
        ]
    )

    t_ecg = transform(ecg)
    assert t_ecg.shape[0] == num_channels
    assert t_ecg.shape[1] == num_samples


@pytest.mark.parametrize("num_channels", [1, 2])
def test_compose_many(num_channels):
    num_augmented_samples = 4
    ecg = nk.ecg_simulate(duration=dur, sampling_rate=sr)
    transform = ComposeMany(
        [
            RandomCrop(num_samples),
            Reverse(),
        ],
        num_augmented_samples=num_augmented_samples,
    )

    t_ecg = transform(ecg)
    assert t_ecg.shape[0] == num_augmented_samples
    assert t_ecg.shape[1] == num_channels
    assert t_ecg.shape[2] == num_samples

    for n in range(1, num_augmented_samples):
        assert torch.all(t_ecg[0].eq(t_ecg[n])) == False
