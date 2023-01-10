import random, torch
from julius.filters import highpass_filter, lowpass_filter


class FrequencyFilter(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float,
        freq_high: float,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high

    def cutoff_frequency(self, frequency: float) -> float:
        return frequency / self.sample_rate

    def sample_uniform_frequency(self):
        return random.uniform(self.freq_low, self.freq_high)


class HighPassFilter(FrequencyFilter):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float = 200,
        freq_high: float = 1200,
    ):
        super().__init__(sample_rate, freq_low, freq_high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency = self.sample_uniform_frequency()
        cutoff = self.cutoff_frequency(frequency)
        return highpass_filter(x, cutoff=cutoff)


class LowPassFilter(FrequencyFilter):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float = 2200,
        freq_high: float = 4000,
    ):
        super().__init__(sample_rate, freq_low, freq_high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency = self.sample_uniform_frequency()
        cutoff = self.cutoff_frequency(frequency)
        return lowpass_filter(x, cutoff=cutoff)


class HighLowPass(torch.nn.Module):
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
