import random, torch
import numpy as np


class GaussianNoise(torch.nn.Module):
    def __init__(self, min_snr=0.01, max_snr=0.5):
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


class RandWanderer(torch.nn.Module):
    def __init__(self, amp, start_phase, end_phase):
        super().__init__()
        self.amp = amp
        self.start_phase = start_phase
        self.end_phase = end_phase

    def forward(self, x):
        sn = np.linspace(self.start_phase, self.end_phase, x.shape[-1])
        sn = self.amp * np.sin(np.pi * sn / 180)
        gauss_noise = GaussianNoise()(x).numpy()
        return x + sn.astype(np.float32) + gauss_noise.astype(np.float32)
