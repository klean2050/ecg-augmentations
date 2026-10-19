"""Plot lead II of a synthetic 12-lead ECG before and after each augmentation.

Usage: python examples/plot_augmentations.py [output.png]
"""

import sys
import matplotlib.pyplot as plt
import torch

import ecg_augmentations as ea

FS = 500
ORIGINAL, AUGMENTED, INK, SURFACE = "#b4b3ad", "#2a78d6", "#3d3d3a", "#fcfcfb"

ecg, peaks = ea.synthetic_ecg(duration=4, fs=FS, n_leads=12, heart_rate=72, seed=0)
detector = lambda signal, fs: peaks.numpy()
noise_bank = ea.MuscleNoise(FS, snr_db=0)(
    torch.zeros(4, 20 * FS) + 1e-3 * torch.randn(4, 20 * FS)
)

transforms = {
    "GaussianNoise(snr_db=15)": ea.GaussianNoise(15),
    "MuscleNoise(snr_db=10)": ea.MuscleNoise(FS, 10),
    "PowerlineNoise(snr_db=10)": ea.PowerlineNoise(FS, snr_db=10),
    "RecordedNoise(snr_db=10)": ea.RecordedNoise(noise_bank, 10),
    "BaselineWander": ea.BaselineWander(FS, amplitude=1.0, freq=0.3),
    "BaselineShift": ea.BaselineShift(amplitude=1.0),
    "HighPassFilter(1 Hz)": ea.HighPassFilter(FS, 1.0),
    "LowPassFilter(15 Hz)": ea.LowPassFilter(FS, 15.0),
    "FrequencyMask(width=10 Hz)": ea.FrequencyMask(FS, width=10, max_freq=20),
    "AmplitudeScale(0.6)": ea.AmplitudeScale(0.6),
    "Invert": ea.Invert(),
    "Reverse": ea.Reverse(),
    "TimeStretch(1.2)": ea.TimeStretch(1.2),
    "RandomTimeWarp(sigma=0.5)": ea.RandomTimeWarp(0.5),
    "TimeShift(0.3 s)": ea.TimeShift(FS, 0.3),
    "Permute(n_patches=4)": ea.Permute(4),
    "RandomResizedCrop(scale=0.6)": ea.RandomResizedCrop(4 * FS, 0.6),
    "TimeMask(2 x 0.4 s)": ea.TimeMask(FS, 0.4, n_masks=2),
    "PRMask": ea.PRMask(FS, 1.0, detector=detector),
    "QRSMask": ea.QRSMask(FS, 1.0, detector=detector),
    "QTMask": ea.QTMask(FS, 1.0, detector=detector),
    "LeadMask(n_leads=11)": ea.LeadMask(11),
    "LeadSubset(lead I only)": ea.LeadSubset(subsets=("1",)),
    "VCGRotation(angle=30)": ea.VCGRotation(angle=(30, 30), scale=1.0),
}


def main(path="docs/images/augmentations.png"):
    lead = 1
    t = torch.arange(ecg.shape[-1]) / FS
    fig, axes = plt.subplots(6, 4, figsize=(14, 12), sharex=True, facecolor=SURFACE)
    for ax, (name, transform) in zip(axes.flat, transforms.items()):
        y = transform(ecg, generator=torch.Generator().manual_seed(3))
        ax.plot(t, ecg[lead], color=ORIGINAL, lw=1.2)
        ax.plot(t, y[lead], color=AUGMENTED, lw=1.0)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        ax.set_yticks([])
        ax.tick_params(colors=INK, labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(ORIGINAL)
    for ax in axes.flat[len(transforms) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=9, color=INK)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in (ORIGINAL, AUGMENTED)]
    fig.legend(
        handles, ["original", "augmented"], loc="upper right", frameon=False, ncol=2
    )
    fig.suptitle("Lead II of a synthetic ECG at 500 Hz", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110, facecolor=SURFACE)


if __name__ == "__main__":
    main(*sys.argv[1:])
