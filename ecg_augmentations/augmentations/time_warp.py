import torch.nn as nn
import numpy as np


class DynamicTimeWarp(nn.Module):
    """
    Stretch and squeeze signal randomly along time axis
    """

    def __init__(self, warps=3, radius=10, step=2):
        super().__init__()
        self.warps = warps
        self.radius = radius
        self.step = step
        self.min_center = self.radius * (self.step + 1)

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, _ = data.shape
        for _ in range(self.warps):
            center = np.random.randint(
                self.min_center, timesteps - self.min_center - self.step
            )
            data = squeeze(data, center, self.radius, self.step)
            data = refill(data, center, self.radius, self.step)
            data = interpolate(data, np.inf)
        return data, label


class ChannelResize(Transformation):
    """Scale amplitude of sample (per channel) by random factor in given magnitude range"""

    def __init__(self, magnitude_range=(0.5, 2)):
        super(ChannelResize, self).__init__(magnitude_range=magnitude_range)
        self.log_magnitude_range = np.log(magnitude_range)

    def __call__(self, sample):
        data, label = sample
        timesteps, channels = data.shape
        resize_factors = np.exp(
            np.random.uniform(*self.log_magnitude_range, size=channels)
        )
        resize_factors_same_shape = np.tile(resize_factors, timesteps).reshape(
            data.shape
        )
        data = np.multiply(resize_factors_same_shape, data)
        return data, label


class TimeOut(Transformation):
    """replace random crop by zeros"""

    def __init__(self, crop_ratio_range=[0.0, 0.5]):
        super(TimeOut, self).__init__(crop_ratio_range=crop_ratio_range)
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, sample):
        data, label = sample
        data = data.copy()
        timesteps, channels = data.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio * timesteps)
        start_idx = random.randint(0, timesteps - crop_timesteps - 1)
        data[start_idx : start_idx + crop_timesteps, :] = 0
        return data, label


class TGaussianBlur1d(Transformation):
    def __init__(self):
        super(TGaussianBlur1d, self).__init__()
        self.conv = torch.nn.modules.conv.Conv1d(1, 1, 5, 1, 2, bias=False)
        self.conv.weight.data = torch.nn.Parameter(
            torch.tensor([[[0.1, 0.2, 0.4, 0.2, 0.1]]])
        )
        self.conv.weight.requires_grad = False

    def __call__(self, sample):
        data, label = sample
        transposed = data.T
        transposed = torch.unsqueeze(transposed, 1)
        blurred = self.conv(transposed)
        return blurred.reshape(data.T.shape).T, label

    def __str__(self):
        return "GaussianBlur"


def interpolate(data, marker):
    timesteps, channels = data.shape
    data = data.flatten(order="F")
    data[data == marker] = np.interp(
        np.where(data == marker)[0], np.where(data != marker)[0], data[data != marker]
    )
    data = data.reshape(timesteps, channels, order="F")
    return data


def squeeze(arr, center, radius, step):
    squeezed = arr[center - step * radius : center + step * radius + 1 : step, :].copy()
    arr[center - step * radius : center + step * radius + 1, :] = np.inf
    arr[center - radius : center + radius + 1, :] = squeezed
    return arr


def refill(arr, center, radius, step):
    left_fill_values = arr[
        center - radius * step - radius : center - radius * step, :
    ].copy()
    right_fill_values = arr[
        center + radius * step + 1 : center + radius * step + radius + 1, :
    ].copy()
    arr[center - radius * step - radius : center - radius * step, :] = arr[
        center + radius * step + 1 : center + radius * step + radius + 1, :
    ] = np.inf
    arr[center - radius * step - radius : center - radius : step, :] = left_fill_values
    arr[
        center + radius + step : center + radius * step + radius + step : step, :
    ] = right_fill_values
    return arr
