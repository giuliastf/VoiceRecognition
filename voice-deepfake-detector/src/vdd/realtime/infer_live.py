from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from typing import Deque

import numpy as np
import sounddevice as sd
import torch

from vdd.data.transforms import build_transform
from vdd.models.cnn import SimpleCNN
from vdd.realtime.mic_stream import MicStream, StreamConfig
from vdd.utils import load_config


def ema_update(prev: float, new: float, alpha: float) -> float:
    return alpha * prev + (1.0 - alpha) * new


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time microphone deepfake detection")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--window", type=float, default=4.0, help="Window size in seconds")
    parser.add_argument("--hop", type=float, default=0.5, help="Hop size in seconds")
    parser.add_argument("--ema", type=float, default=0.8, help="EMA smoothing factor")
    parser.add_argument("--threshold", type=float, default=0.6, help="Fake probability threshold")
    parser.add_argument("--playback", action="store_true", help="Enable playback hotkey")
    parser.add_argument("--playback-seconds", type=float, default=3.0, help="Playback duration")
    parser.add_argument("--silence-rms", type=float, default=0.01, help="RMS threshold for silence")
    parser.add_argument("--bias", type=float, default=0.0, help="Bias added to raw probability")
    args = parser.parse_args()

    config = load_config(args.config)

    sample_rate = config.get("dataset", "sample_rate")
    n_mels = config.get("features", "n_mels")
    n_fft = config.get("features", "n_fft")
    hop_length = config.get("features", "hop_length")

    transform = build_transform(
        sample_rate,
        int(args.window),
        n_mels,
        n_fft,
        hop_length,
        augment=None,
        is_train=False,
    )

    model_channels = config.get("model", "channels", default=[16, 32, 64, 128])
    model_dropout = config.get("model", "dropout", default=0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(in_channels=1, channels=model_channels, dropout=model_dropout).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    stream_cfg = StreamConfig(
        sample_rate=sample_rate,
        chunk_size=int(sample_rate * 0.1),
        buffer_seconds=max(args.window, args.playback_seconds) + 1.0,
    )
    mic = MicStream(stream_cfg)
    mic.start()

    smoothed = 0.5
    recent: Deque[float] = deque(maxlen=5)
    playback_requested = threading.Event()
    stop_requested = threading.Event()

    def key_listener() -> None:
        if os.name == "nt":
            import msvcrt

            while not stop_requested.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getwch().lower()
                    if key == "p":
                        playback_requested.set()
                    if key == "q":
                        stop_requested.set()
                        break
                time.sleep(0.05)
        else:
            import select

            while not stop_requested.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable:
                    key = sys.stdin.read(1).lower()
                    if key == "p":
                        playback_requested.set()
                    if key == "q":
                        stop_requested.set()
                        break

    if args.playback:
        listener = threading.Thread(target=key_listener, daemon=True)
        listener.start()

    print("Listening... Press 'p' to playback, 'q' to quit.")
    try:
        while not stop_requested.is_set():
            time.sleep(args.hop)
            window = mic.read_last(args.window)
            rms = float(np.sqrt(np.mean(window**2))) if window.size else 0.0
            if rms < args.silence_rms:
                decision = "SILENCE"
                bar = "-" * 20
                print(f"RMS={rms:.4f} [{bar}] {decision}   ", end="\r", flush=True)
                continue
            waveform = torch.from_numpy(window).unsqueeze(0)
            with torch.no_grad():
                mel = transform(waveform, sample_rate).unsqueeze(0).to(device)
                logit = model(mel)
                prob = torch.sigmoid(logit).item() + args.bias
                prob = max(0.0, min(1.0, prob))

            recent.append(prob)
            smoothed = ema_update(smoothed, prob, args.ema)
            decision = "FAKE" if smoothed >= args.threshold else "REAL"
            bar = "#" * int(smoothed * 20)
            bar = bar.ljust(20, "-")
            print(f"RMS={rms:.4f} P(fake)={smoothed:.3f} [{bar}] {decision}   ", end="\r", flush=True)

            if args.playback and playback_requested.is_set():
                playback_requested.clear()
                playback = mic.read_last(args.playback_seconds)
                sd.play(playback, sample_rate)
                sd.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_requested.set()
        mic.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
