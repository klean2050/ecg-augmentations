import numpy as np
import pytest, torch
from pathlib import Path

import ecg_augmentations as ea


@pytest.mark.parametrize("shape", [(1, -1), (12, -1), (3, 12, -1)])
def test_shape_preserved(transform, ecg, shape):
    x = ecg[0] if shape == (1, -1) else ecg.expand(*shape[:-1], -1).clone()
    y = transform(x)
    assert y.shape[:-1] == x.shape[:-1]
    assert y.shape[-1] == (
        2000
        if isinstance(transform, (ea.RandomCrop, ea.RandomResizedCrop))
        else x.shape[-1]
    )
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_preserved(transform, ecg, dtype):
    assert transform(ecg.to(dtype)).dtype == dtype


def test_numpy_roundtrip(transform, ecg):
    x = ecg.double().numpy()
    y = transform(x, generator=torch.Generator().manual_seed(0))
    assert isinstance(y, np.ndarray) and y.dtype == np.float64
    expected = transform(ecg.double(), generator=torch.Generator().manual_seed(0))
    np.testing.assert_allclose(y, expected.numpy())


def test_input_not_modified(transform, ecg):
    x = ecg.clone()
    transform(x)
    assert torch.equal(x, ecg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda(transform, ecg):
    y = transform(ecg.cuda())
    assert y.is_cuda


def test_invalid_inputs():
    with pytest.raises(TypeError):
        ea.Invert()(torch.ones(2, 10, dtype=torch.long))
    with pytest.raises(ValueError):
        ea.Invert()(torch.ones(2, 2, 2, 10))


def test_compose_many(ecg):
    aug = ea.Compose([ea.RandomCrop(500), ea.GaussianNoise()])
    views = ea.ComposeMany(aug, num_augmented_samples=4)(ecg)
    assert views.shape == (4, 12, 500)
    assert not torch.equal(views[0], views[1])
    assert ea.ComposeMany([ea.Invert()], 2)(ecg.numpy()).shape == (2, 12, 2500)


def test_return_mask(ecg):
    y, mask = ea.TimeMask(250, duration=0.5, return_mask=True)(ecg)
    assert mask.shape == ecg.shape and mask.dtype == torch.bool
    assert torch.equal(y[mask], torch.zeros(int(mask.sum())))
    assert torch.equal(y[~mask], ecg[~mask])
    _, mask = ea.LeadMask(n_leads=3, return_mask=True)(ecg)
    assert mask.all(dim=-1).sum() == 3


def test_differentiable(ecg):
    x = ecg.clone().requires_grad_()
    aug = ea.Compose(
        [
            ea.GaussianNoise(),
            ea.AmplitudeScale(),
            ea.LowPassFilter(250),
            ea.TimeStretch(),
        ]
    )
    aug(x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_real_ecg_fixture():
    x = torch.load(Path(__file__).parent / "some_ecg.pt")
    aug = ea.Compose(
        [
            ea.RandomCrop(500),
            ea.GaussianNoise(),
            ea.BaselineWander(100),
            ea.TimeMask(100),
        ]
    )
    assert aug(x).shape == (1, 500)
