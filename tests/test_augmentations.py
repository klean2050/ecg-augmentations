import torch, pytest
import neurokit2 as nk

from ecg_augmentations import *


dur, sr = 10, 100
num_samples = dur * sr


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("p", (0, 1))
def test_random_apply(p, seed):
    torch.manual_seed(seed)
    transform = RandomApply([Invert()], p=p)

    ecg = nk.ecg_simulate(duration=dur, sampling_rate=sr)
    t_ecg = transform(ecg)
    if p == 0:
        assert torch.eq(t_ecg, ecg).all() == True
    elif p == 1:
        assert torch.eq(t_ecg, ecg).all() == False
