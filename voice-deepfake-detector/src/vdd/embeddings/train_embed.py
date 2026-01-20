from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Set

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from vdd.embeddings.utils import extract_embeddings, load_ids
from vdd.utils import ensure_dir, load_config, set_seed


def build_ids_path(manifests_dir: Path, split: str, split_mode: str) -> Path:
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




def main() -> None:
    parser = argparse.ArgumentParser(description="Train embedding-based classifier")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("project", "seed", default=42))

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
    max_train = config.get("embeddings", "max_train_samples", default=None)
    max_val = config.get("embeddings", "max_val_samples", default=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(embedding_model)
    model = Wav2Vec2Model.from_pretrained(embedding_model, use_safetensors=True).to(device)
    model.eval()

    train_ids = set(load_ids(build_ids_path(manifests_dir, "train", split_mode)))
    val_ids = set(load_ids(build_ids_path(manifests_dir, "val", split_mode)))

    train_generators = None
    val_generators = None
    if split_mode == "crossgen":
        train_generators = set(config.get("splits", "crossgen_train_generators"))
        val_generators = set(config.get("splits", "crossgen_train_generators"))

    train_embeds, train_labels = extract_embeddings(
        dataset_name=dataset_name,
        split="train",
        audio_column=audio_column,
        audio_id_column=audio_id_column,
        class_column=class_column,
        real_label=real_label,
        include_audio_ids=train_ids,
        include_generators=train_generators,
        streaming=streaming,
        sample_rate=sample_rate,
        processor=processor,
        model=model,
        device=device,
        batch_size=batch_size,
        max_samples=max_train,
    )

    val_embeds, val_labels = extract_embeddings(
        dataset_name=dataset_name,
        split="train",
        audio_column=audio_column,
        audio_id_column=audio_id_column,
        class_column=class_column,
        real_label=real_label,
        include_audio_ids=val_ids,
        include_generators=val_generators,
        streaming=streaming,
        sample_rate=sample_rate,
        processor=processor,
        model=model,
        device=device,
        batch_size=batch_size,
        max_samples=max_val,
    )

    np.save(runs_dir / "train_embeddings.npy", train_embeds)
    np.save(runs_dir / "train_labels.npy", train_labels)
    np.save(runs_dir / "val_embeddings.npy", val_embeds)
    np.save(runs_dir / "val_labels.npy", val_labels)

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(train_embeds, train_labels)
    joblib.dump(clf, runs_dir / "classifier.joblib")

    val_preds = clf.predict(val_embeds)
    val_accuracy = float((val_preds == val_labels).mean())
    metrics_path = runs_dir / "val_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"val_accuracy": val_accuracy}, handle, indent=2)

    print({"val_accuracy": val_accuracy})


if __name__ == "__main__":
    main()
