import pytest, torch
from ecg_augmentations.utils import *


@pytest.mark.parametrize(
    "tensor,expected_value",
    [
        (torch.zeros(1), False),
        (torch.zeros(1, 1000), False),
        (torch.zeros(16, 1000), False),
        (torch.zeros(1, 1, 1000), True),
        (torch.zeros(16, 1, 1000), True),
    ],
)
def test_has_valid_batch_dim(tensor, expected_value):

    assert has_valid_batch_dim(tensor) == expected_value


def test_add_batch_dim():
    tensor = torch.ones(1, 48000)
    expected_tensor = torch.ones(1, 1, 48000)

    tensor = add_batch_dim(tensor)
    assert torch.eq(tensor, expected_tensor).all()
    assert has_valid_batch_dim(tensor) == True

    tensor = torch.ones(48000)
    expected_tensor = torch.ones(1, 48000)

    tensor = add_batch_dim(tensor)
    assert torch.eq(tensor, expected_tensor).all()
    assert has_valid_batch_dim(tensor) == False


def test_remove_batch_dim():
    tensor = torch.ones(1, 1, 48000)
    expected_tensor = torch.ones(1, 48000)

    tensor = remove_batch_dim(tensor)
    assert torch.eq(tensor, expected_tensor).all()

    tensor = torch.ones(1, 48000)
    expected_tensor = torch.ones(48000)

    tensor = remove_batch_dim(tensor)
    assert torch.eq(tensor, expected_tensor).all()
    assert has_valid_batch_dim(tensor) == False
