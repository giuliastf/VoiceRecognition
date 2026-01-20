from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import soundfile as sf
import numpy as np
import torch
import torchaudio
from datasets import Audio, load_dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def load_audio_sample(audio: dict, target_sample_rate: int) -> torch.Tensor:
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

    if sample_rate is None:
        sample_rate = target_sample_rate

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    return waveform.squeeze(0)


def stream_hf_dataset(
    dataset_name: str,
    split: str,
    audio_column: str,
    audio_id_column: str,
    class_column: str,
    include_audio_ids: Set[str],
    include_generators: Optional[Set[str]],
    real_label: str,
    streaming: bool,
) -> Iterable[dict]:
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    dataset = dataset.cast_column(audio_column, Audio(decode=False))

    def allowed_generator(class_label: str, real_label_value: str) -> bool:
        if include_generators is None:
            return True
        if class_label == real_label_value:
            return True
        return class_label in include_generators

    return (
        sample
        for sample in dataset
        if sample[audio_id_column] in include_audio_ids
        and allowed_generator(sample[class_column], real_label_value=real_label)
    )


def batch_iter(items: List[Tuple[str, torch.Tensor, int]], batch_size: int) -> Iterable[List[Tuple[str, torch.Tensor, int]]]:
    batch: List[Tuple[str, torch.Tensor, int]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def label_from_class(label: str, real_label: str) -> int:
    return 0 if label == real_label else 1


def extract_embeddings(
    dataset_name: str,
    split: str,
    audio_column: str,
    audio_id_column: str,
    class_column: str,
    real_label: str,
    include_audio_ids: Set[str],
    include_generators: Optional[Set[str]],
    streaming: bool,
    sample_rate: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
    batch_size: int,
    max_samples: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    samples: List[Tuple[str, torch.Tensor, int]] = []
    dataset_iter = stream_hf_dataset(
        dataset_name=dataset_name,
        split=split,
        audio_column=audio_column,
        audio_id_column=audio_id_column,
        class_column=class_column,
        include_audio_ids=include_audio_ids,
        include_generators=include_generators,
        real_label=real_label,
        streaming=streaming,
    )

    collected = 0
    for sample in dataset_iter:
        waveform = load_audio_sample(sample[audio_column], sample_rate)
        label = label_from_class(sample[class_column], real_label)
        samples.append((sample[audio_id_column], waveform, label))
        collected += 1
        if max_samples is not None and collected >= max_samples:
            break

    embeddings = []
    labels = []
    for batch in batch_iter(samples, batch_size):
        waveforms = [item[1].cpu().numpy().astype("float32") for item in batch]
        labels.extend([item[2] for item in batch])
        inputs = processor(waveforms, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1)
        embeddings.append(pooled.cpu().numpy())

    if not embeddings:
        return np.array([]), np.array([])
    return np.concatenate(embeddings), np.array(labels)
