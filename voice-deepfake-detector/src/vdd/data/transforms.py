from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio


@dataclass
class AudioTransformConfig:
    sample_rate: int
    clip_seconds: int
    n_mels: int
    n_fft: int
    hop_length: int

    @property
    def clip_samples(self) -> int:
        return self.sample_rate * self.clip_seconds


@dataclass
class AugmentConfig:
    noise_std: float
    gain_min: float
    gain_max: float


class AudioToLogMel:
    def __init__(self, config: AudioTransformConfig, augment: Optional[AugmentConfig], is_train: bool) -> None:
        self.config = config
        self.augment = augment
        self.is_train = is_train
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def _apply_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.is_train or self.augment is None:
            return waveform
        gain = torch.empty(1).uniform_(self.augment.gain_min, self.augment.gain_max).item()
        waveform = waveform * gain
        if self.augment.noise_std > 0:
            noise = torch.randn_like(waveform) * self.augment.noise_std
            waveform = waveform + noise
        return waveform

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.config.sample_rate:
            return waveform
        return torchaudio.functional.resample(waveform, sample_rate, self.config.sample_rate)

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        target = self.config.clip_samples
        current = waveform.shape[-1]
        if current == target:
            return waveform
        if current > target:
            return waveform[..., :target]
        pad = target - current
        return torch.nn.functional.pad(waveform, (0, pad))

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        waveform = self._resample(waveform, sample_rate)
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self._apply_augment(waveform)
        waveform = self._pad_or_trim(waveform)
        mel = self.mel(waveform)
        log_mel = self.amplitude_to_db(mel)
        return log_mel


def build_transform(
    sample_rate: int,
    clip_seconds: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    augment: Optional[AugmentConfig],
    is_train: bool,
) -> AudioToLogMel:
    config = AudioTransformConfig(
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return AudioToLogMel(config, augment=augment, is_train=is_train)
