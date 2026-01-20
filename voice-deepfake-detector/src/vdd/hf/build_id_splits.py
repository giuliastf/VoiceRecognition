from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

from vdd.utils import ensure_dir, load_config, set_seed


def collect_audio_ids(
    dataset_name: str, audio_id_column: str, streaming: bool, max_ids: int | None
) -> List[str]:
    if streaming:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    else:
        try:
            dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=False,
                verification_mode="no_checks",
            )
        except TypeError:
            dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=False,
                ignore_verifications=True,
            )
    if not streaming:
        audio_ids = list(set(dataset[audio_id_column]))
        return audio_ids

    seen = set()
    for sample in dataset:
        audio_id = sample[audio_id_column]
        seen.add(audio_id)
        if max_ids is not None and len(seen) >= max_ids:
            break
    return list(seen)


def split_ids(ids: List[str], train_ratio: float, val_ratio: float) -> Dict[str, List[str]]:
    total = len(ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def save_ids(ids: List[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in ids:
            handle.write(f"{item}\n")


def save_split_files(manifests_dir: Path, splits: Dict[str, List[str]]) -> None:
    save_ids(splits["train"], manifests_dir / "train_ids.txt")
    save_ids(splits["val"], manifests_dir / "val_ids.txt")
    save_ids(splits["test"], manifests_dir / "test_ids.txt")


def save_crossgen_files(
    manifests_dir: Path,
    train_ids: List[str],
    test_ids: List[str],
    train_generators: List[str],
    test_generators: List[str],
) -> None:
    save_ids(train_ids, manifests_dir / "crossgen_train_ids.txt")
    save_ids(test_ids, manifests_dir / "crossgen_test_ids.txt")
    save_ids(train_generators, manifests_dir / "crossgen_train_generators.txt")
    save_ids(test_generators, manifests_dir / "crossgen_test_generators.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leak-free audio_id splits")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("project", "seed", default=42)
    set_seed(seed)

    dataset_name = config.get("dataset", "name")
    audio_id_column = config.get("dataset", "audio_id_column")

    train_ratio = config.get("splits", "train_ratio")
    val_ratio = config.get("splits", "val_ratio")

    train_generators = config.get("splits", "crossgen_train_generators")
    test_generators = config.get("splits", "crossgen_test_generators")

    manifests_dir = ensure_dir(config.get("paths", "manifests_dir"))

    use_streaming = config.get("splits", "use_streaming", default=False)
    max_ids = config.get("splits", "max_audio_ids", default=None)

    ids = collect_audio_ids(dataset_name, audio_id_column, use_streaming, max_ids)
    ids.sort()
    random.shuffle(ids)
    splits = split_ids(ids, train_ratio, val_ratio)

    save_split_files(manifests_dir, splits)
    save_crossgen_files(
        manifests_dir,
        splits["train"],
        splits["test"],
        train_generators,
        test_generators,
    )

    print(f"Saved splits to {manifests_dir}")


if __name__ == "__main__":
    main()
