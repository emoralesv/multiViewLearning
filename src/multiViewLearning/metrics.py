from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import f1_score
import torch


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def f1_top1(logits: torch.Tensor, targets: torch.Tensor, average: str = "macro") -> float:
    preds = logits.argmax(dim=1).cpu().numpy()
    y = targets.cpu().numpy()
    return float(f1_score(y, preds, average=average))

