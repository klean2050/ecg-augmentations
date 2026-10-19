import ecg_augmentations as ea

FS = 250
ECG, PEAKS = ea.synthetic_ecg(duration=10, fs=FS, n_leads=12, seed=0)


def detector(signal, fs):
    return PEAKS[PEAKS < len(signal)].numpy()


TRANSFORMS = {
    "RandomCrop": lambda: ea.RandomCrop(duration=8, fs=FS),
    "HighPassFilter": lambda: ea.HighPassFilter(FS),
    "LowPassFilter": lambda: ea.LowPassFilter(FS),
    "Invert": lambda: ea.Invert(),
    "Reverse": lambda: ea.Reverse(),
    "Permute": lambda: ea.Permute(),
    "AmplitudeScale": lambda: ea.AmplitudeScale(per_lead=True),
    "GaussianNoise": lambda: ea.GaussianNoise(),
    "BaselineWander": lambda: ea.BaselineWander(FS),
    "PowerlineNoise": lambda: ea.PowerlineNoise(FS, harmonics=2),
    "TimeMask": lambda: ea.TimeMask(FS, n_masks=(1, 3)),
    "LeadMask": lambda: ea.LeadMask(n_leads=(1, 6)),
    "PRMask": lambda: ea.PRMask(FS, detector=detector),
    "QRSMask": lambda: ea.QRSMask(FS, detector=detector),
    "QTMask": lambda: ea.QTMask(FS, detector=detector),
    "TimeStretch": lambda: ea.TimeStretch(),
}
