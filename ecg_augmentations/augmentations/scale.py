import torch.nn as nn
import numpy as np, torch


class Stretch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return NotImplementedError

    def forward(self, x):
        return NotImplementedError


class Compress(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return NotImplementedError

    def forward(self, x):
        return NotImplementedError


class Scale(torch.nn.Module):
    def __init__(self, max_factor=5):
        super().__init__()
        self.factor = max_factor * 10

    def forward(self, x):
        factor = np.random.randint(self.factor) / 10
        return torch.mul(factor, x)


class TimeWarp(nn.Module):
    """
    Stretch and squeeze signal randomly along the time axis
    """

    def __init__(self, warps=3, radius=10, step=2):
        super().__init__()
        self.warps = warps
        self.radius = radius
        self.step = step
        self.min_center = self.radius * (self.step + 1)

    def forward(self, x):
        for _ in range(self.warps):
            self.center = np.random.randint(
                self.min_center, len(x) - self.min_center - self.step
            )
            x = self.squeeze(x, self.center, self.radius, self.step)
            x = self.refill(x, self.center, self.radius, self.step)
            x = self.interpolate(x, np.inf)
        return x

    def interpolate(self, x, marker):
        timesteps, channels = data.shape
        data = data.flatten(order="F")
        data[data == marker] = np.interp(
            np.where(data == marker)[0],
            np.where(data != marker)[0],
            data[data != marker],
        )
        data = data.reshape(timesteps, channels, order="F")
        return data

    def squeeze(self, x):
        step_start = self.center - (self.step * self.radius)
        step_end = self.center + (self.step * self.radius) + 1
        start = self.center - self.radius
        end = self.center + self.radius + 1

        squeezed = x[step_start : step_end : self.step, :].copy()
        x[step_start : step_end + 1, :] = np.inf
        x[start:end, :] = squeezed
        return x

    def refill(self, x):
        step_start = self.center - (self.step * self.radius)
        step_end = self.center + (self.step * self.radius) + 1
        temp = self.center + self.radius + self.step

        left_fill_values = x[step_start - self.radius : step_start, :].copy()
        right_fill_values = x[step_end : step_end + self.radius, :].copy()

        x[step_start - self.radius : step_start, :] = np.inf
        x[step_end : step_end + self.radius, :] = np.inf
        x[step_start - self.radius : step_start, :] = left_fill_values
        x[temp : temp + (self.radius * self.step) : self.step, :] = right_fill_values
        return x
