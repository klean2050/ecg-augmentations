import random
import torch
import augment


class PitchShift:
    def __init__(self, audio_length, sr, pitch_cents_min=700, pitch_cents_max=700):
        self.audio_length = audio_length
        self.sr = sr
        self.pitch_cents_min = pitch_cents_min
        self.pitch_cents_max = pitch_cents_max
        self.src_info = {"rate": self.sr}
        self.target_info = {
            "channels": 1,
            "length": self.audio_length,
            "rate": self.sr,
        }

    def __call__(self, audio):
        n_steps = random.randint(self.pitch_cents_min, self.pitch_cents_max)
        effect_chain = augment.EffectChain().pitch(n_steps).rate(self.sr)

        y = effect_chain.apply(
            audio, src_info=self.src_info, target_info=self.target_info
        )

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return audio.clone()

        if y.shape[1] != audio.shape[1]:
            if y.shape[1] > audio.shape[1]:
                y = y[:, audio.shape[1]]
            else:
                y0 = torch.zeros(1, audio.shape[1])
                y0[:, :y.shape[1]] = y
                y = y0
        return y