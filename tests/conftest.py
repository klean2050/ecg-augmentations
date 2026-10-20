import pytest, torch

from tests.common import ECG, TRANSFORMS


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture
def ecg():
    return ECG.clone()


@pytest.fixture(params=sorted(TRANSFORMS))
def transform(request):
    return TRANSFORMS[request.param]()
