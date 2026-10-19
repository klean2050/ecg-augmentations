import pytest, torch

import ecg_augmentations as ea


def test_generator_reproducible(transform, ecg):
    y1 = transform(ecg, generator=torch.Generator().manual_seed(1))
    y2 = transform(ecg, generator=torch.Generator().manual_seed(1))
    assert torch.equal(y1, y2)


def test_global_seed_reproducible(transform, ecg):
    torch.manual_seed(1)
    y1 = transform(ecg)
    torch.manual_seed(1)
    assert torch.equal(y1, transform(ecg))


@pytest.mark.parametrize(
    "name", ["GaussianNoise", "AmplitudeScale", "TimeMask", "TimeStretch", "RandomCrop"]
)
def test_seeds_differ(name, ecg):
    from tests.common import TRANSFORMS

    transform = TRANSFORMS[name]()
    y1 = transform(ecg, generator=torch.Generator().manual_seed(1))
    y2 = transform(ecg, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(y1, y2)


def test_per_sample_parameters(ecg):
    batch = ecg.expand(8, -1, -1)
    y = ea.Compose([ea.AmplitudeScale(), ea.RandomCrop(1000)])(batch)
    assert len({tuple(s[0, :5].tolist()) for s in y}) == 8


def test_probability(ecg):
    batch = ecg[:1].expand(4000, -1, -1)
    assert torch.equal(ea.Invert(p=0)(batch), batch)
    assert torch.equal(ea.Invert(p=1)(batch), -batch)
    applied = (ea.Invert(p=0.5)(batch) != batch).any(dim=-1).float().mean()
    assert abs(applied - 0.5) < 0.03
    applied = (
        (ea.RandomApply([ea.Invert()], p=0.3)(batch) != batch)
        .any(dim=-1)
        .float()
        .mean()
    )
    assert abs(applied - 0.3) < 0.03


def test_one_of(ecg):
    batch = ecg[:1].expand(3000, -1, -1)
    y = ea.OneOf([ea.Invert(), ea.AmplitudeScale(0.5)])(batch)
    inverted = (y == -batch).all(dim=-1).float().mean()
    halved = (y == 0.5 * batch).all(dim=-1).float().mean()
    assert abs(inverted - 0.5) < 0.04 and inverted + halved == 1
