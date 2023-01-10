import torch


class Invert(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.neg(x)
