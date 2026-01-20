from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vdd.data.collate import collate_mels
from vdd.data.dataset import WaveFakeStreamingDataset
from vdd.data.transforms import AugmentConfig, build_transform
from vdd.metrics import compute_metrics
from vdd.models.cnn import SimpleCNN
from vdd.utils import Config, ensure_dir, load_config, set_seed


def load_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _ids_path(manifests_dir: Path, split: str, split_mode: str) -> Path:
    if split_mode == "crossgen":
        if split == "train":
            return manifests_dir / "crossgen_train_ids.txt"
        if split == "val":
            return manifests_dir / "val_ids.txt"
        return manifests_dir / "crossgen_test_ids.txt"
    if split == "train":
        return manifests_dir / "train_ids.txt"
    if split == "val":
        return manifests_dir / "val_ids.txt"
    return manifests_dir / "test_ids.txt"


def build_dataloader(config: Config, split: str, generators: List[str] | None, split_mode: str) -> DataLoader:
    dataset_name = config.get("dataset", "name")
    streaming = config.get("dataset", "streaming", default=False)
    audio_column = config.get("dataset", "audio_column")
    audio_id_column = config.get("dataset", "audio_id_column")
    class_column = config.get("dataset", "class_column")
    real_label = config.get("dataset", "real_label")
    fake_labels = set(config.get("dataset", "fake_labels"))

    sample_rate = config.get("dataset", "sample_rate")
    clip_seconds = config.get("dataset", "clip_seconds")
    n_mels = config.get("features", "n_mels")
    n_fft = config.get("features", "n_fft")
    hop_length = config.get("features", "hop_length")
    augment_enabled = config.get("augment", "enabled", default=False)

    manifests_dir = Path(config.get("paths", "manifests_dir"))
    cache_dir = Path(config.get("paths", "cache_dir")) if config.get("cache", "enabled") else None

    ids_path = _ids_path(manifests_dir, split, split_mode)
    if split == "train":
        max_samples = config.get("training", "max_train_samples_per_epoch")
    elif split == "val":
        max_samples = config.get("training", "max_val_samples")
    else:
        max_samples = None

    include_audio_ids = set(load_ids(ids_path))
    is_train = split == "train"
    augment = None
    if is_train and augment_enabled:
        augment = AugmentConfig(
            noise_std=config.get("augment", "noise_std", default=0.0),
            gain_min=config.get("augment", "gain_min", default=1.0),
            gain_max=config.get("augment", "gain_max", default=1.0),
        )
        cache_dir = None

    transform = build_transform(sample_rate, clip_seconds, n_mels, n_fft, hop_length, augment, is_train)

    dataset = WaveFakeStreamingDataset(
        dataset_name=dataset_name,
        split="train",
        streaming=streaming,
        audio_column=audio_column,
        audio_id_column=audio_id_column,
        class_column=class_column,
        real_label=real_label,
        fake_labels=fake_labels,
        include_audio_ids=include_audio_ids,
        include_generators=set(generators) if generators else None,
        transform=transform,
        cache_dir=cache_dir,
        max_samples=max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=config.get("training", "batch_size"),
        num_workers=config.get("training", "num_workers"),
        collate_fn=collate_mels,
    )


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    steps = 0
    for mels, labels in tqdm(loader, desc="train", leave=False):
        mels = mels.to(device)
        labels = labels.to(device)
        logits = model(mels)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        steps += 1
    return running_loss / max(steps, 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for mels, labels in tqdm(loader, desc="eval", leave=False):
            mels = mels.to(device)
            logits = model(mels)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    if not all_preds:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    return compute_metrics(labels, preds)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def plot_training_curves(history: List[Dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_f1 = [item["val_f1"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, val_f1, marker="o", color="tab:orange")
    axes[1].set_title("Validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("project", "seed", default=42))

    experiment_name = config.get("experiment", "name")
    split_mode = config.get("experiment", "split")
    runs_dir = ensure_dir(Path(config.get("paths", "runs_dir")) / experiment_name)

    train_generators = None
    val_generators = None
    if split_mode == "crossgen":
        train_generators = config.get("splits", "crossgen_train_generators")
        val_generators = config.get("splits", "crossgen_train_generators")

    train_loader = build_dataloader(config, "train", train_generators, split_mode)
    val_loader = build_dataloader(config, "val", val_generators, split_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_channels = config.get("model", "channels", default=[16, 32, 64, 128])
    model_dropout = config.get("model", "dropout", default=0.0)
    model = SimpleCNN(in_channels=1, channels=model_channels, dropout=model_dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("training", "learning_rate"),
        weight_decay=config.get("training", "weight_decay"),
    )

    best_f1 = 0.0
    patience = config.get("training", "early_stopping_patience")
    patience_left = patience

    history: List[Dict[str, float]] = []
    for epoch in range(1, config.get("training", "epochs") + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
        )
        metrics_path = runs_dir / f"metrics_epoch_{epoch}.json"
        save_metrics({"train_loss": train_loss, **val_metrics}, metrics_path)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_left = patience
            torch.save(model.state_dict(), runs_dir / "best_model.pt")
        else:
            patience_left -= 1

        print(f"Epoch {epoch}: loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f}")
        if patience_left <= 0:
            print("Early stopping triggered")
            break

    torch.save(model.state_dict(), runs_dir / "final_model.pt")
    save_metrics({"best_val_f1": best_f1}, runs_dir / "summary.json")
    save_metrics({"history": history}, runs_dir / "history.json")
    plot_training_curves(history, runs_dir / "training_curves.png")


if __name__ == "__main__":
    main()
