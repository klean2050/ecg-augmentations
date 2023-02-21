# PyTorch ECG Augmentations

ECG time-series augmentations library for PyTorch. The focus of this repo is to:
- Provide many transformations in an easy-to-use Python interface.
- Easily control stochastic / sequential transformations.
- Make every transformation differentiable through `torch.nn.Module`.
- Optimize ECG transformations for CPU and GPU devices.

This repo supports stochastic transformations as used often in self-supervised and semi-supervised learning methods. One can apply a single stochastic augmentation or create as many stochastically transformed ECG examples from a single interface. This package follows the conventions set out by [torchaudio-augmentations](https://github.com/Spijkervet/torchaudio-augmentations), with an ECG sample defined as a tensor of `[lead, time]`, or a batched representation `[batch, lead, time]`. Each individual augmentation can be initialized on its own, or be wrapped around a `RandomApply` interface which will apply the augmentation with probability `p`.

**Note**: Current version has been tested on single-lead ECG. The repo is in beta, we appreciate any contributions.

## Installation

Just clone the repository and install the library:
```
git clone https://github.com/klean2050/ecg-augmentations
pip install -e ecg-augmentations
```

## Usage

One can define a single or several augmentations, which are applied sequentially to an ECG sample:

```python
from ecg_augmentations import *

# 1 lead, 100 samples
ecg = torch.load("tests/some_ecg.pt")

num_samples = ecg.shape[-1] * 0.5
transforms = [
    RandomCrop(n_samples=num_samples),
    RandomApply([PRMask()], p=0.4),
    RandomApply([QRSMask()], p=0.4),
    RandomApply([Scale()], p=0.5),
    RandomApply([Permute()], p=0.6),
    GaussianNoise(max_snr=0.005),
    RandomApply([Invert()], p=0.2),
    RandomApply([Reverse()], p=0.2),
]
```

One can also define a stochastic augmentation on multiple transformations:

```python
transforms = [
    RandomCrop(n_samples=num_samples),
    RandomApply([PRMask(), QRSMask()], p=0.8),
    RandomApply([Invert(), Reverse()], p=0.5)
]
```

One can return either one ...

```python
transform = Compose(transforms=transforms)
transformed_ecg =  transform(ecg)
>> transformed_ecg.shape = [num_leads, num_samples]
```

... or many versions of the same example:

```python
transform = ComposeMany(transforms=transforms, num_augmented_samples=2)
transformed_ecg =  transform(ecg)
>> transformed_ecg.shape = [2, num_leads, num_samples]
```

## Citation

* Kleanthis Avramidis, PhD Student in Computer Science

University of Southern California, Citation TBD