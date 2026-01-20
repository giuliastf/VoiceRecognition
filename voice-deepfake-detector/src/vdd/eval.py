from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc

from vdd.data.collate import collate_mels
from vdd.data.dataset import WaveFakeStreamingDataset
from vdd.data.transforms import build_transform
from vdd.metrics import compute_confusion, compute_metrics
from vdd.models.cnn import SimpleCNN
from vdd.utils import Config, ensure_dir, load_config


def load_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _ids_path(manifests_dir: Path, split: str, split_mode: str) -> Path:
    if split_mode == "crossgen":
        if split == "test":
            return manifests_dir / "crossgen_test_ids.txt"
        return manifests_dir / "val_ids.txt"
    if split == "test":
        return manifests_dir / "test_ids.txt"
    return manifests_dir / "val_ids.txt"


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

    manifests_dir = Path(config.get("paths", "manifests_dir"))
    cache_dir = Path(config.get("paths", "cache_dir")) if config.get("cache", "enabled") else None

    ids_path = _ids_path(manifests_dir, split, split_mode)

    include_audio_ids = set(load_ids(ids_path))
    transform = build_transform(sample_rate, clip_seconds, n_mels, n_fft, hop_length, None, False)

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
        max_samples=None,
    )

    return DataLoader(
        dataset,
        batch_size=config.get("training", "batch_size"),
        num_workers=config.get("training", "num_workers"),
        collate_fn=collate_mels,
    )


def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for mels, labels in loader:
            mels = mels.to(device)
            logits = model(mels)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    preds = np.concatenate(all_preds) if all_preds else np.array([])
    probs = np.concatenate(all_probs) if all_probs else np.array([])
    return labels, probs, preds


def save_confusion(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["real", "fake"])
    ax.set_yticklabels(["real", "fake"])
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_roc_curve(labels: np.ndarray, probs: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_pr_curve(labels: np.ndarray, probs: np.ndarray, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"AP={avg_precision:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config.get("experiment", "name")
    split_mode = config.get("experiment", "split")
    runs_dir = ensure_dir(Path(config.get("paths", "runs_dir")) / experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_channels = config.get("model", "channels", default=[16, 32, 64, 128])
    model_dropout = config.get("model", "dropout", default=0.0)
    model = SimpleCNN(in_channels=1, channels=model_channels, dropout=model_dropout).to(device)
    model_path = runs_dir / "best_model.pt"
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    generators = None
    if split_mode == "crossgen":
        generators = config.get("splits", "crossgen_test_generators")

    test_loader = build_dataloader(config, "test", generators, split_mode)
    labels_arr, probs_arr, preds_arr = collect_predictions(model, test_loader, device)

    metrics = compute_metrics(labels_arr, preds_arr) if labels_arr.size else {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    confusion, _ = compute_confusion(labels_arr, preds_arr)
    save_confusion(confusion, runs_dir / "confusion_matrix.png")
    if labels_arr.size:
        save_roc_curve(labels_arr, probs_arr, runs_dir / "roc_curve.png")
        save_pr_curve(labels_arr, probs_arr, runs_dir / "pr_curve.png")

    metrics_path = runs_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
