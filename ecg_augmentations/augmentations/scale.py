import torch.nn as nn
import cv2, math
import numpy as np, torch


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
    adapted from https://github.com/pritamqu/SSL-ECG/blob/master/implementation/signal_transformation_task.py
    """

    def __init__(self, sr, warps, stretch_factor, squeeze_factor):
        super().__init__()
        self.sr = sr
        self.warps = warps
        self.stretch_factor = stretch_factor
        self.squeeze_factor = squeeze_factor

    def forward(self, x):
        x = x.numpy()
        total_time = x.shape[-1] // self.sr
        segment_time = total_time / self.warps
        sequence = list(range(self.warps))

        stretch = np.random.choice(
            sequence, math.ceil(len(sequence) / 2), replace=False
        )
        squeeze = list(set(sequence).difference(set(stretch)))

        initialize = True
        for i in sequence:
            orig_signal = x[
                ...,
                int(i * np.floor(segment_time * self.sr)) : int(
                    (i + 1) * np.floor(segment_time * self.sr)
                ),
            ]
            orig_signal = orig_signal.reshape(-1, 1)

            if i in stretch:
                output_shape = int(
                    np.ceil(np.shape(orig_signal)[0] * self.stretch_factor)
                )
                new_signal = cv2.resize(
                    orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR
                )
                if initialize == True:
                    time_warped = new_signal
                    initialize = False
                else:
                    time_warped = np.vstack((time_warped, new_signal))
            elif i in squeeze:
                output_shape = int(
                    np.ceil(np.shape(orig_signal)[0] * self.squeeze_factor)
                )
                new_signal = cv2.resize(
                    orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR
                )
                if initialize == True:
                    time_warped = new_signal
                    initialize = False
                else:
                    time_warped = np.vstack((time_warped, new_signal))

        return torch.Tensor(time_warped).reshape(1, -1)
