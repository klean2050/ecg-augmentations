import torch


def has_valid_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    return True if tensor.ndim == 3 else False


def add_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(dim=0)


def remove_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.squeeze(dim=0)
