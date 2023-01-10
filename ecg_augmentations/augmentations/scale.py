import torch


class Stretch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flip(x, dims=[-1])


class Compress(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flip(x, dims=[-1])


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flip(x, dims=[-1])
