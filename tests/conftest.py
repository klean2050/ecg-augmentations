import pytest

from tests.common import ECG, TRANSFORMS


@pytest.fixture
def ecg():
    return ECG.clone()


@pytest.fixture(params=sorted(TRANSFORMS))
def transform(request):
    return TRANSFORMS[request.param]()
