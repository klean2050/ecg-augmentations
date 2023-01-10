import torch


class GetNeighbor(torch.nn.Module):
    """
    Return a neighboring segment to input
    from the same subject recordings
    """

    def __init__(self, depth=None):
        super().__init__()
        self.depth = depth
        return NotImplementedError

    def forward(self, x):
        return NotImplementedError
