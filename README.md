# PyTorch ECG Augmentations

ECG time-series augmentations library for PyTorch. The focus of this repo is to:
- Provide many transformations in an easy-to-use Python interface.
- Easily control stochastic / sequential transformations.
- Make every transformation differentiable through `nn.Module`.
- Optimise ECG transformations for CPU and GPU devices.

This repo supports stochastic transformations as used often in self-supervised and semi-supervised learning methods. One can apply a single stochastic augmentation or create as many stochastically transformed ECG examples from a single interface. This package follows the conventions set out by `torchaudio-augmentations`, with an ECG sample defined as a tensor of `[lead, time]`, or a batched representation `[batch, lead, time]`. Each individual augmentation can be initialized on its own, or be wrapped around a `RandomApply` interface which will apply the augmentation with probability `p`. **Note**: Current version has been solely tested on single-lead ECG samples.

## Usage

One can define a single or several augmentations, which are applied sequentially to an ECG sample.

```python
from ecg_augmentations import *

ecg = torch.load("tests/some_ecg.pt")

num_samples = ecg.shape[-1] * 0.5
transforms = [
    RandomResizedCrop(n_samples=int(num_samples)),
    RandomApply([PolarityInversion()], p=0.8),
    RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
    HighLowPass(sample_rate=sr), # this will always be applied
    RandomApply([Delay(sample_rate=sr)], p=0.5),
]
```

One can also define a stochastic augmentation on multiple transformations:

```python
transforms = [
    RandomResizedCrop(n_samples=num_samples),
    RandomApply([PolarityInversion(), Noise(min_snr=0.001, max_snr=0.005)], p=0.8),
    RandomApply([Delay(sample_rate=sr), Reverb(sample_rate=sr)], p=0.5)
]
```

We can return either one or many versions of the same audio example:
```python
transform = Compose(transforms=transforms)
transformed_ecg =  transform(ecg)
>> transformed_ecg.shape = [num_leads, num_samples]
```

```python
transform = ComposeMany(transforms=transforms, num_augmented_samples=2)
transformed_ecg =  transform(ecg)
>> transformed_ecg.shape = [2, num_leads, num_samples]
```

# Citation

TBD