"""Plot lead II of a synthetic 12-lead ECG before and after representative augmentations.

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
transforms = {
    "GaussianNoise(snr_db=15)": ea.GaussianNoise(15),
    "MuscleNoise(snr_db=10)": ea.MuscleNoise(FS, 10),
    "PowerlineNoise(snr_db=10)": ea.PowerlineNoise(FS, snr_db=10),
    "BaselineWander": ea.BaselineWander(FS, amplitude=1.0, freq=0.3),
    "LowPassFilter(15 Hz)": ea.LowPassFilter(FS, 15.0),
    "AmplitudeScale(0.6)": ea.AmplitudeScale(0.6),
    "TimeStretch(1.2)": ea.TimeStretch(1.2),
    "RandomTimeWarp(sigma=0.5)": ea.RandomTimeWarp(0.5),
    "Permute(n_patches=4)": ea.Permute(4),
    "TimeMask(2 × 0.4 s)": ea.TimeMask(FS, 0.4, n_masks=2),
    "QRSMask": ea.QRSMask(FS, 1.0, detector=detector),
    "VCGRotation(angle=30)": ea.VCGRotation(angle=(30, 30), scale=1.0),
}


def main(path="docs/images/augmentations.png"):
    lead = 1
    t = torch.arange(ecg.shape[-1]) / FS
    fig, axes = plt.subplots(4, 3, figsize=(12, 7.5), sharex=True, facecolor=SURFACE)
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
