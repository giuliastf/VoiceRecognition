from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import soundfile as sf
import torch
from torch.utils.data import IterableDataset

from datasets import Audio

from vdd.hf.hf_stream import stream_dataset, take_samples
from vdd.data.transforms import AudioToLogMel


class WaveFakeStreamingDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        streaming: bool,
        audio_column: str,
        audio_id_column: str,
        class_column: str,
        real_label: str,
        fake_labels: Set[str],
        include_audio_ids: Set[str],
        include_generators: Optional[Set[str]] = None,
        transform: Optional[AudioToLogMel] = None,
        cache_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.streaming = streaming
        self.audio_column = audio_column
        self.audio_id_column = audio_id_column
        self.class_column = class_column
        self.real_label = real_label
        self.fake_labels = fake_labels
        self.include_audio_ids = include_audio_ids
        self.include_generators = include_generators
        self.transform = transform
        self.cache_dir = cache_dir
        self.max_samples = max_samples

    def _label_from_class(self, label: str) -> int:
        return 0 if label == self.real_label else 1

    def _cache_path(self, audio_id: str, class_label: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe_label = class_label.replace("/", "_")
        suffix = ""
        if self.transform is not None and hasattr(self.transform, "config"):
            config = self.transform.config
            suffix = f"_sr{config.sample_rate}_c{config.clip_seconds}_m{config.n_mels}_fft{config.n_fft}_hop{config.hop_length}"
        return self.cache_dir / f"{audio_id}_{safe_label}{suffix}.pt"

    def _load_or_compute_mel(self, audio: dict, audio_id: str, class_label: str) -> torch.Tensor:
        cache_path = self._cache_path(audio_id, class_label)
        if cache_path and cache_path.exists():
            try:
                return torch.load(cache_path, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(cache_path, map_location="cpu")

        if "array" in audio and audio["array"] is not None:
            waveform = torch.tensor(audio["array"]).float()
            sample_rate = audio.get("sampling_rate")
        else:
            if audio.get("bytes") is not None:
                data, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            else:
                data, sample_rate = sf.read(audio["path"], dtype="float32")
            waveform = torch.from_numpy(data)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.transpose(0, 1)

        if sample_rate is None and self.transform is not None:
            sample_rate = self.transform.config.sample_rate

        if sample_rate is None:
            raise ValueError("Sample rate missing for audio decode")

        if self.transform is None:
            raise ValueError("Transform is required to compute log-mel features")

        mel = self.transform(waveform, sample_rate)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mel, cache_path)
        return mel

    def _should_include_generator(self, class_label: str) -> bool:
        if self.include_generators is None:
            return True
        if class_label == self.real_label:
            return True
        return class_label in self.include_generators

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, int]]:
        dataset = stream_dataset(self.dataset_name, split=self.split, streaming=self.streaming)
        dataset = dataset.cast_column(self.audio_column, Audio(decode=False))
        filtered = (
            sample
            for sample in dataset
            if sample[self.audio_id_column] in self.include_audio_ids
            and self._should_include_generator(sample[self.class_column])
        )
        for sample in take_samples(filtered, self.max_samples):
            class_label = sample[self.class_column]
            audio_id = sample[self.audio_id_column]
            audio = sample[self.audio_column]
            mel = self._load_or_compute_mel(audio, audio_id, class_label)
            label = self._label_from_class(class_label)
            yield mel, label
