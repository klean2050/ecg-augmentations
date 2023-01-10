import random, torch.nn as nn
from ecg_augmentations import HighPassFilter, LowPassFilter


class HighLowPass(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        lowpass_freq_low: float = 2200,
        lowpass_freq_high: float = 4000,
        highpass_freq_low: float = 200,
        highpass_freq_high: float = 1200,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        self.high_pass_filter = HighPassFilter(
            sample_rate, highpass_freq_low, highpass_freq_high
        )
        self.low_pass_filter = LowPassFilter(
            sample_rate, lowpass_freq_low, lowpass_freq_high
        )

    def forward(self, x):
        return (
            self.low_pass_filter(x)
            if random.randint(0, 1)
            else self.high_pass_filter(x)
        )
