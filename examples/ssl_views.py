"""Create two augmented views per ECG for contrastive self-supervised learning."""

import torch
from torch.utils.data import DataLoader, Dataset

import ecg_augmentations as ea

FS = 500

augment = ea.Compose(
    [
        ea.RandomCrop(duration=8, fs=FS),
        ea.RandomApply([ea.BaselineWander(FS), ea.GaussianNoise((15, 30))], p=0.8),
        ea.PowerlineNoise(FS, freq=(49.5, 50.5), p=0.3),
        ea.AmplitudeScale((0.7, 1.4), p=0.5),
        ea.TimeStretch((0.9, 1.1), p=0.5),
        ea.OneOf([ea.TimeMask(FS, (0.2, 0.8)), ea.LeadMask((1, 6))], p=0.5),
    ]
)


class SyntheticECG(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 64

    def __getitem__(self, i):
        ecg, _ = ea.synthetic_ecg(duration=10, fs=FS, heart_rate=50 + i, seed=i)
        return self.transform(ecg) if self.transform else ecg


if __name__ == "__main__":
    # Option 1: augment per sample in DataLoader workers.
    views = ea.ComposeMany(augment, num_augmented_samples=2)
    loader = DataLoader(SyntheticECG(views), batch_size=16, num_workers=2)
    for batch in loader:
        v1, v2 = batch.unbind(dim=1)
        print("per-sample views:", tuple(v1.shape), tuple(v2.shape))
        break

    # Option 2: augment whole batches, e.g. on the GPU; parameters are still per sample.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in DataLoader(SyntheticECG(), batch_size=16):
        batch = batch.to(device)
        v1, v2 = augment(batch), augment(batch)
        print("batch views on", device, tuple(v1.shape))
        break
