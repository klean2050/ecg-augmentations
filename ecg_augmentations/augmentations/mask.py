import torch, numpy as np
import neurokit2 as nk


class PRMask(torch.nn.Module):
    def __init__(self, sr, ratio=0.5):
        super().__init__()
        self.sr = sr
        self.ratio = ratio

    def forward(self, x):
        x = x.clone()
        # detect R peaks
        signals, _ = nk.ecg_peaks(x.numpy()[0], sampling_rate=self.sr)
        r_peaks = np.where(signals["ECG_R_Peaks"])[0]
        # P peaks happen ~0.2 sec before R ones
        duration = self.sr // 5
        intervals = [torch.arange(ri - duration, ri) for ri in r_peaks]

        for interval in intervals:
            interval = interval[interval < x.shape[-1]]
            if torch.rand(1) > self.ratio:
                x[..., interval] = (
                    x[..., interval[0] - 1].clone()
                    if interval[0]
                    else x[..., 0].clone()
                )
        return x


class QRSMask(torch.nn.Module):
    def __init__(self, sr, ratio=0.5):
        super().__init__()
        self.sr = sr
        self.ratio = ratio

    def forward(self, x):
        x = x.clone()
        # detect R peaks
        signals, _ = nk.ecg_peaks(x.numpy()[0], sampling_rate=self.sr)
        r_peaks = np.where(signals["ECG_R_Peaks"])[0]
        # Q peaks happen ~0.05 sec before R
        # S peaks happen ~0.05 sec after R
        duration = self.sr // 20
        intervals = [torch.arange(ri - duration, ri + duration) for ri in r_peaks]

        for interval in intervals:
            interval = interval[interval < x.shape[-1]]
            if torch.rand(1) > self.ratio:
                x[..., interval] = (
                    x[..., interval[0] - 1].clone()
                    if interval[0]
                    else x[..., 0].clone()
                )
        return x


class QTMask(torch.nn.Module):
    def __init__(self, sr, ratio=0.5):
        super().__init__()
        self.sr = sr
        self.ratio = ratio

    def forward(self, x):
        x = x.clone()
        # detect R peaks
        signals, _ = nk.ecg_peaks(x.numpy()[0], sampling_rate=self.sr)
        r_peaks = np.where(signals["ECG_R_Peaks"])[0]
        # T peaks happen ~0.3 sec after R
        dur1, dur2 = self.sr // 20, self.sr // 3
        intervals = [torch.arange(ri - dur1, ri + dur2) for ri in r_peaks]

        for interval in intervals:
            interval = interval[interval < x.shape[-1]]
            if torch.rand(1) > self.ratio:
                x[..., interval] = (
                    x[..., interval[0] - 1].clone()
                    if interval[0]
                    else x[..., 0].clone()
                )
        return x


class RandMask(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        x = x.clone()
        intervals, durations = [], []
        min_win, max_win = 0, 0.05 * x.shape[-1]

        def cap(a, b):
            return [i for i in a if i in b]

        while sum(durations) < self.ratio * x.shape[-1]:
            random_start = np.random.randint(0, x.shape[-1] - max_win)
            random_end = random_start + np.random.randint(min_win, max_win)
            random_win = np.arange(random_start, random_end)

            intersections = [len(cap(p, random_win)) for p in intervals]
            if sum(intersections) >= random_end - random_start:
                continue

            intervals.append(random_win)
            durations.append(random_end - random_start - sum(intersections))

        for interval in intervals:
            x[..., interval] = (
                x[..., interval[0] - 1].clone() if interval[0] else x[..., 0].clone()
            )
        return x
