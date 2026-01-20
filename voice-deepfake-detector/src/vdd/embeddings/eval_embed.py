from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Set

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from vdd.embeddings.utils import extract_embeddings, load_ids
from vdd.utils import ensure_dir, load_config


def build_ids_path(manifests_dir: Path, split: str, split_mode: str) -> Path:
    if split_mode == "crossgen":
        if split == "test":
            return manifests_dir / "crossgen_test_ids.txt"
        return manifests_dir / "val_ids.txt"
    if split == "test":
        return manifests_dir / "test_ids.txt"
    return manifests_dir / "val_ids.txt"


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
    parser = argparse.ArgumentParser(description="Evaluate embedding-based classifier")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config.get("experiment", "name")
    split_mode = config.get("experiment", "split")
    runs_dir = ensure_dir(Path(config.get("paths", "runs_dir")) / experiment_name)

    dataset_name = config.get("dataset", "name")
    streaming = config.get("dataset", "streaming", default=False)
    audio_column = config.get("dataset", "audio_column")
    audio_id_column = config.get("dataset", "audio_id_column")
    class_column = config.get("dataset", "class_column")
    real_label = config.get("dataset", "real_label")
    sample_rate = config.get("dataset", "sample_rate")

    manifests_dir = Path(config.get("paths", "manifests_dir"))
    embedding_model = config.get("embeddings", "model_name", default="facebook/wav2vec2-base")
    batch_size = config.get("embeddings", "batch_size", default=8)
    max_test = config.get("embeddings", "max_test_samples", default=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(embedding_model)
    model = Wav2Vec2Model.from_pretrained(embedding_model, use_safetensors=True).to(device)
    model.eval()

    test_ids = set(load_ids(build_ids_path(manifests_dir, "test", split_mode)))
    test_generators: Optional[Set[str]] = None
    if split_mode == "crossgen":
        test_generators = set(config.get("splits", "crossgen_test_generators"))

    test_embeds, test_labels = extract_embeddings(
        dataset_name=dataset_name,
        split="train",
        audio_column=audio_column,
        audio_id_column=audio_id_column,
        class_column=class_column,
        real_label=real_label,
        include_audio_ids=test_ids,
        include_generators=test_generators,
        streaming=streaming,
        sample_rate=sample_rate,
        processor=processor,
        model=model,
        device=device,
        batch_size=batch_size,
        max_samples=max_test,
    )

    clf = joblib.load(runs_dir / "classifier.joblib")
    probs = clf.predict_proba(test_embeds)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "precision": float(precision_score(test_labels, preds, zero_division=0)),
        "recall": float(recall_score(test_labels, preds, zero_division=0)),
        "f1": float(f1_score(test_labels, preds, zero_division=0)),
    }

    confusion = np.array(
        [
            [int(((test_labels == 0) & (preds == 0)).sum()), int(((test_labels == 0) & (preds == 1)).sum())],
            [int(((test_labels == 1) & (preds == 0)).sum()), int(((test_labels == 1) & (preds == 1)).sum())],
        ]
    )

    save_confusion(confusion, runs_dir / "confusion_matrix.png")
    save_roc_curve(test_labels, probs, runs_dir / "roc_curve.png")
    save_pr_curve(test_labels, probs, runs_dir / "pr_curve.png")

    with (runs_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
