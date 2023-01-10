import random
import numpy as np
import torch


class Noise(torch.nn.Module):
    def __init__(self, min_snr=0.0001, max_snr=0.01):
        """
        min_snr: Minimum signal-to-noise ratio
        max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, x):
        std = torch.std(x)
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)
        noise = np.random.normal(0.0, noise_std, size=x.shape)
        return x + noise.astype(np.float32)
