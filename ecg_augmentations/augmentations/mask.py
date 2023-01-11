import torch, numpy as np
import neurokit2 as nk


class PRMask(torch.nn.Module):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        # detect R peaks
        signals, _ = nk.ecg_peaks(x.numpy(), sampling_rate=100)
        r_peaks = np.where(signals["ECG_R_Peaks"])[0]
        # P peaks happen ~20 samples before R
        # P peak detection has weaker accuracy
        intervals = [torch.arange(ri - 20, ri + 1) for ri in r_peaks]
        for interval in intervals:
            interval = interval[interval < len(x)]
            if torch.rand(1) > self.ratio:
                x[interval] = (
                    x[interval[0] - 1].clone() if interval[0] else x[0].clone()
                )
        return x


class QRSMask(torch.nn.Module):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        # detect R peaks
        signals, _ = nk.ecg_peaks(x.numpy(), sampling_rate=100)
        r_peaks = np.where(signals["ECG_R_Peaks"])[0]
        # Q peaks happen ~5 samples before R
        # S peaks happen ~5 samples after R
        intervals = [torch.arange(ri - 5, ri + 6) for ri in r_peaks]
        for interval in intervals:
            interval = interval[interval < len(x)]
            if torch.rand(1) > self.ratio:
                x[interval] = (
                    x[interval[0] - 1].clone() if interval[0] else x[0].clone()
                )
        return x


class QTMask(torch.nn.Module):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        # detect R peaks
        signals, _ = nk.ecg_peaks(x.numpy(), sampling_rate=100)
        r_peaks = np.where(signals["ECG_R_Peaks"])[0]
        # Q peaks happen ~5 samples before R
        # T peaks happen ~30 samples after R
        intervals = [torch.arange(ri - 5, ri + 31) for ri in r_peaks]
        for interval in intervals:
            interval = interval[interval < len(x)]
            if torch.rand(1) > self.ratio:
                x[interval] = (
                    x[interval[0] - 1].clone() if interval[0] else x[0].clone()
                )
        return x


class RandMask(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        intervals, durations = [], []
        min_win, max_win = 0 * len(x), 0.05 * len(x)

        def cap(a, b):
            return [i for i in a if i in b]

        while sum(durations) < self.ratio * len(x):
            random_start = np.random.randint(0, len(x) - max_win)
            random_end = random_start + np.random.randint(min_win, max_win)
            random_win = np.arange(random_start, random_end)

            intersections = [len(cap(p, random_win)) for p in intervals]
            if sum(intersections) >= random_end - random_start:
                continue

            intervals.append(random_win)
            durations.append(random_end - random_start - sum(intersections))

        for interval in intervals:
            x[interval] = x[interval[0] - 1].clone() if interval[0] else x[0].clone()
        return x
