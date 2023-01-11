import random, torch.nn as nn


class RandomCrop(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, x):
        max_samples = x.shape[-1]
        start_idx = random.randint(0, max_samples - self.n_samples)
        return x[..., start_idx : start_idx + self.n_samples]
