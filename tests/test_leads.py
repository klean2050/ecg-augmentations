import pytest, torch

import ecg_augmentations as ea
from ecg_augmentations.augmentations.leads import INDEPENDENT, KORS, LEAD_SETS, LEADS


def vcg(x):
    return torch.tensor(KORS, dtype=x.dtype) @ x[..., INDEPENDENT, :]


def test_synthetic_limb_leads_consistent(ecg):
    assert torch.allclose(ea.DeriveLimbLeads()(ecg), ecg, atol=1e-6)


def test_derive_limb_leads():
    x = torch.randn(12, 100, dtype=torch.float64)
    y = ea.DeriveLimbLeads()(x)
    i, ii, iii, avr, avl, avf = y[:6]
    assert torch.allclose(iii, ii - i) and torch.allclose(
        avr + avl + avf, torch.zeros(100, dtype=torch.float64)
    )
    assert torch.equal(y[[0, 1, *range(6, 12)]], x[[0, 1, *range(6, 12)]])


def test_vcg_rotation_identity(ecg):
    assert torch.allclose(ea.VCGRotation(angle=0, scale=1)(ecg), ecg, atol=1e-5)


def test_vcg_rotation_transforms_vcg(ecg):
    x = ecg.double()
    y = ea.VCGRotation(angle=(30, 30), scale=1)(x)
    r = ea.augmentations.leads.rotation(torch.full((1, 3), 30.0).deg2rad()).double()[0]
    assert torch.allclose(vcg(y), r @ vcg(x), atol=1e-8)
    assert torch.allclose(vcg(y).norm(dim=0), vcg(x).norm(dim=0), atol=1e-8)
    assert torch.allclose(ea.DeriveLimbLeads()(y), y)
    z = ea.VCGRotation(angle=0, scale=2)(x)
    assert torch.allclose(vcg(z), 2 * vcg(x), atol=1e-8)


def test_vcg_rotation_preserves_timing(ecg):
    y = ea.VCGRotation(angle=20)(ecg)
    assert torch.equal(y.abs().sum(0).argmax(), ecg.abs().sum(0).argmax())


def test_lead_subset(ecg):
    y = ea.LeadSubset(subsets=("4",))(ecg)
    kept = [LEADS.index(lead) for lead in LEAD_SETS["4"]]
    masked = [i for i in range(12) if i not in kept]
    assert torch.equal(y[kept], ecg[kept]) and y[masked].eq(0).all()
    sets = {
        tuple((s != 0).any(-1).tolist())
        for s in ea.LeadSubset()(ecg.expand(200, -1, -1))
    }
    assert len(sets) == 5


@pytest.mark.parametrize(
    "make",
    [lambda: ea.VCGRotation(), lambda: ea.LeadSubset(), lambda: ea.DeriveLimbLeads()],
)
def test_requires_12_leads(make):
    with pytest.raises(ValueError):
        make()(torch.zeros(8, 100))


def test_unknown_lead_set():
    with pytest.raises(ValueError):
        ea.LeadSubset(subsets=("5",))
