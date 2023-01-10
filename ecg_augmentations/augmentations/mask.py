import torch


class PRMask(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flip(x, dims=[-1])


class QRSMask(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flip(x, dims=[-1])


class QTMask(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flip(x, dims=[-1])


class RandMask(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flip(x, dims=[-1])
