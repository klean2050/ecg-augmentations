import random, torch
import numpy as np


class Permute(torch.nn.Module):
    def __init__(self, n_patches=10):
        super().__init__()
        self.n_patches = n_patches

    def forward(self, x):
        n_bps = self.n_patches - 1
        base = np.arange(1, len(x) - 2)
        bps = np.random.choice(base, n_bps, replace=False)
        bps = np.sort(bps)

        intervals = [torch.arange(bps[0])]
        for this_bps in bps[1:]:
            last_item = intervals[-1][-1]
            intervals.append(torch.arange(last_item + 1, this_bps))
        intervals.append(torch.arange(bps[-1], len(x)))

        random.shuffle(intervals)
        permuted = [x[i].clone() for i in intervals]
        return torch.hstack(permuted)
