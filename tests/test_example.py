import neurokit2 as nk
from ecg_augmentations import *


def test_example():
    dur, sr = 10, 100
    ecg = nk.ecg_simulate(duration=dur, sampling_rate=sr)
    transforms = [
        RandomCrop(n_samples=dur * sr),
        RandomApply([Invert()], p=0.8),
        RandomApply([GaussianNoise(min_snr=0.3, max_snr=0.5)], p=0.3),
        RandomApply([HighLowPass(sample_rate=sr)], p=0.8),
        RandomApply([Reverse()], p=0.3),
    ]

    transform = Compose(transforms=transforms)
    transformed_audio = transform(ecg)
    assert transformed_audio.shape[0] == 1
