from __future__ import annotations

from typing import Iterable, Optional

from datasets import load_dataset


def stream_dataset(dataset_name: str, split: str = "train", streaming: bool = True) -> Iterable[dict]:
    return load_dataset(dataset_name, split=split, streaming=streaming)


def take_samples(dataset: Iterable[dict], max_samples: Optional[int]) -> Iterable[dict]:
    if max_samples is None:
        yield from dataset
        return
    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break
        yield sample
