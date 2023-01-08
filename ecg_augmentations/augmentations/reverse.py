import torch


class Reverse(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flip(x, dims=[-1])
