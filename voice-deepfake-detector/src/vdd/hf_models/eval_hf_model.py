from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Audio, load_dataset
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from vdd.embeddings.utils import load_audio_sample, load_ids
from vdd.utils import ensure_dir, load_config


def build_ids_path(manifests_dir: Path, split: str, split_mode: str) -> Path:
    if split_mode == "crossgen":
        if split == "test":
            return manifests_dir / "crossgen_test_ids.txt"
        return manifests_dir / "val_ids.txt"
    if split == "test":
        return manifests_dir / "test_ids.txt"
    return manifests_dir / "val_ids.txt"


def allowed_generator(class_label: str, real_label: str, include_generators: Optional[Set[str]]) -> bool:
    if include_generators is None:
        return True
    if class_label == real_label:
        return True
    return class_label in include_generators


def iter_samples(
    dataset_name: str,
    split: str,
    streaming: bool,
    audio_column: str,
    audio_id_column: str,
    class_column: str,
    include_audio_ids: Set[str],
    include_generators: Optional[Set[str]],
    real_label: str,
) -> Iterable[dict]:
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    dataset = dataset.cast_column(audio_column, Audio(decode=False))
    for sample in dataset:
        if sample[audio_id_column] not in include_audio_ids:
            continue
        if not allowed_generator(sample[class_column], real_label, include_generators):
            continue
        yield sample


def infer_fake_score(outputs: List[dict], fake_labels: List[str], real_labels: List[str]) -> float:
    label_to_score = {item["label"].lower(): float(item["score"]) for item in outputs}

    for label in fake_labels:
        if label.lower() in label_to_score:
            return label_to_score[label.lower()]

    for label in real_labels:
        if label.lower() in label_to_score:
            return 1.0 - label_to_score[label.lower()]

    for name, score in label_to_score.items():
        if "fake" in name or "spoof" in name:
            return score
        if "bonafide" in name or "real" in name:
            return 1.0 - score

    return list(label_to_score.values())[0]


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
    parser = argparse.ArgumentParser(description="Evaluate pretrained HF deepfake model")
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
    model_name = config.get("hf_model", "model_name")
    fake_labels = config.get("hf_model", "fake_labels", default=["fake", "spoof"])
    real_labels = config.get("hf_model", "real_labels", default=["real", "bonafide"])
    max_test = config.get("hf_model", "max_test_samples", default=None)

    test_ids = set(load_ids(build_ids_path(manifests_dir, "test", split_mode)))
    test_generators: Optional[Set[str]] = None
    if split_mode == "crossgen":
        test_generators = set(config.get("splits", "crossgen_test_generators"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name, use_safetensors=True).to(device)
    model.eval()
    id2label = model.config.id2label or {}
    with (runs_dir / "id2label.json").open("w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in id2label.items()}, handle, indent=2)

    fake_label_id = config.get("hf_model", "fake_label_id", default=None)
    if fake_label_id is None:
        lower_labels = {idx: str(label).lower() for idx, label in id2label.items()}
        for idx, name in lower_labels.items():
            if "fake" in name or "spoof" in name:
                fake_label_id = idx
                break
        if fake_label_id is None and len(lower_labels) == 2:
            fake_label_id = 1

    expected_sr = getattr(processor, "sampling_rate", None) or sample_rate

    labels: List[int] = []
    probs: List[float] = []
    preds: List[int] = []

    count = 0
    for sample in iter_samples(
        dataset_name=dataset_name,
        split="train",
        streaming=streaming,
        audio_column=audio_column,
        audio_id_column=audio_id_column,
        class_column=class_column,
        include_audio_ids=test_ids,
        include_generators=test_generators,
        real_label=real_label,
    ):
        waveform = load_audio_sample(sample[audio_column], expected_sr).cpu().numpy().astype("float32")
        inputs = processor(waveform, sampling_rate=expected_sr, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs_tensor = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        outputs = []
        for idx, score in enumerate(probs_tensor):
            label = id2label.get(idx, str(idx))
            outputs.append({"label": label, "score": float(score)})
        if fake_label_id is not None and 0 <= fake_label_id < len(probs_tensor):
            fake_score = float(probs_tensor[fake_label_id])
            pred_label = int(np.argmax(probs_tensor))
            pred_value = 1 if pred_label == fake_label_id else 0
        else:
            fake_score = infer_fake_score(outputs, fake_labels, real_labels)
            pred_value = 1 if fake_score >= 0.5 else 0
        labels.append(0 if sample[class_column] == real_label else 1)
        probs.append(fake_score)
        preds.append(pred_value)
        count += 1
        if max_test is not None and count >= max_test:
            break

    labels_arr = np.array(labels)
    probs_arr = np.array(probs)
    preds_arr = np.array(preds) if preds else (probs_arr >= 0.5).astype(int)

    debug = {
        "expected_sample_rate": int(expected_sr),
        "num_samples": int(labels_arr.size),
        "label_distribution": {
            "real": int((labels_arr == 0).sum()),
            "fake": int((labels_arr == 1).sum()),
        },
        "prediction_distribution": {
            "real": int((preds_arr == 0).sum()),
            "fake": int((preds_arr == 1).sum()),
        },
        "mean_fake_score": float(probs_arr.mean()) if probs_arr.size else 0.0,
    }
    with (runs_dir / "debug.json").open("w", encoding="utf-8") as handle:
        json.dump(debug, handle, indent=2)

    metrics = {
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
        "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
    }

    confusion = np.array(
        [
            [int(((labels_arr == 0) & (preds_arr == 0)).sum()), int(((labels_arr == 0) & (preds_arr == 1)).sum())],
            [int(((labels_arr == 1) & (preds_arr == 0)).sum()), int(((labels_arr == 1) & (preds_arr == 1)).sum())],
        ]
    )

    save_confusion(confusion, runs_dir / "confusion_matrix.png")
    save_roc_curve(labels_arr, probs_arr, runs_dir / "roc_curve.png")
    save_pr_curve(labels_arr, probs_arr, runs_dir / "pr_curve.png")

    with (runs_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
