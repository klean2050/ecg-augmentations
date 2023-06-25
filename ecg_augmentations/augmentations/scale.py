import torch.nn as nn
import cv2, math
import numpy as np, torch
from scipy.interpolate import interp1d

class Scale(torch.nn.Module):
    def __init__(self, max_factor=5):
        super().__init__()
        self.factor = max_factor * 10

    def forward(self, x):
        factor = np.random.randint(self.factor) / 10
        return torch.mul(factor, x)


class TimeWarp(nn.Module):
    """Currently supports only stretching"""

    def __init__(self, stretch_factor="random"):
        super().__init__()
        self.ratio = stretch_factor
        if self.ratio == "random":
            self.ratio = np.random.uniform(0.1, 0.9)
        assert self.ratio > 0 and self.ratio < 1, "Stretch factor must be between 0 and 1"

    def forward(self, x):
        x = x.numpy()
        length = len(x[0])

        # Generate a time vector for the original signal
        time_vector = np.linspace(0, length, num=length)
        # Randomly generate a time warping function
        warp = np.linspace(0, length * self.ratio, num=length)

        # Apply time warping to each lead of the ECG signal
        warped_signal = np.zeros_like(x)
        for i in range(len(x)):
            # Interpolate the ECG signal using the warping function
            interpolation_func = interp1d(time_vector, x[i])
            warped_signal[i] = interpolation_func(warp)

        return torch.Tensor(warped_signal)