from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import sounddevice as sd


@dataclass
class StreamConfig:
    sample_rate: int
    chunk_size: int
    buffer_seconds: float

    @property
    def max_samples(self) -> int:
        return int(self.sample_rate * self.buffer_seconds)


class RingBuffer:
    def __init__(self, max_samples: int) -> None:
        self.max_samples = max_samples
        self.buffer: Deque[float] = deque(maxlen=max_samples)
        self.lock = threading.Lock()

    def extend(self, samples: np.ndarray) -> None:
        with self.lock:
            self.buffer.extend(samples.tolist())

    def read_last(self, num_samples: int) -> np.ndarray:
        with self.lock:
            if not self.buffer:
                return np.zeros(num_samples, dtype=np.float32)
            data = list(self.buffer)[-num_samples:]
        if len(data) < num_samples:
            pad = np.zeros(num_samples - len(data), dtype=np.float32)
            return np.concatenate([pad, np.array(data, dtype=np.float32)])
        return np.array(data, dtype=np.float32)


class MicStream:
    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.ring = RingBuffer(config.max_samples)
        self.stream = sd.InputStream(
            samplerate=config.sample_rate,
            channels=1,
            blocksize=config.chunk_size,
            callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            return
        mono = indata[:, 0].astype(np.float32)
        self.ring.extend(mono)

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()

    def read_last(self, seconds: float) -> np.ndarray:
        num_samples = int(self.config.sample_rate * seconds)
        return self.ring.read_last(num_samples)
