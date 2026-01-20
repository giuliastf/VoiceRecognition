from __future__ import annotations

from typing import List, Tuple

import torch


def collate_mels(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    mels, labels = zip(*batch)
    mels_tensor = torch.stack(mels)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return mels_tensor, labels_tensor
