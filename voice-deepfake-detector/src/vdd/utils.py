from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class Config:
    raw: Dict[str, Any]

    def get(self, *keys: str, default: Any = None) -> Any:
        data: Any = self.raw
        for key in keys:
            if not isinstance(data, dict) or key not in data:
                return default
            data = data[key]
        return data


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> Config:
    path_obj = Path(path)
    config = load_yaml(str(path_obj))
    extends = config.get("extends")
    if extends:
        base_path = (path_obj.parent / extends).resolve()
        base = load_yaml(str(base_path))
        config = merge_dicts(base, {k: v for k, v in config.items() if k != "extends"})
    return Config(config)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
