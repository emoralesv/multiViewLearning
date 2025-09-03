from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import DualViewDataset
from .metrics import accuracy_top1, f1_top1


@dataclass
class TrainResult:
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    train_f1: float
    val_f1: float
    gates: Optional[torch.Tensor] = None


def _collate(samples: List) -> Tuple[List[torch.Tensor], torch.Tensor]:
    views = list(zip(*[s.views for s in samples]))  # list per view
    views = [torch.stack(v, dim=0) for v in views]
    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    return views, labels


def train_model(
    model: nn.Module,
    train_ds: DualViewDataset,
    val_ds: DualViewDataset,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TrainResult:
    device_t = torch.device(device)
    model.to(device_t)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    va_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    last_gates: Optional[torch.Tensor] = None
    for _ in range(epochs):
        model.train()
        tr_losses = []
        tr_logits = []
        tr_targets = []
        for xs, y in tr_loader:
            xs = [x.to(device_t) for x in xs]
            y = y.to(device_t)
            optim.zero_grad()
            out = model(xs)
            if isinstance(out, tuple):
                logits, gates = out
                last_gates = gates.detach().cpu() if gates is not None else None
            else:
                logits = out
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())
            tr_logits.append(logits.detach().cpu())
            tr_targets.append(y.detach().cpu())

        model.eval()
        va_losses = []
        va_logits = []
        va_targets = []
        with torch.no_grad():
            for xs, y in va_loader:
                xs = [x.to(device_t) for x in xs]
                y = y.to(device_t)
                out = model(xs)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, y)
                va_losses.append(loss.item())
                va_logits.append(logits.detach().cpu())
                va_targets.append(y.detach().cpu())

    tr_logits_t = torch.cat(tr_logits, dim=0)
    tr_targets_t = torch.cat(tr_targets, dim=0)
    va_logits_t = torch.cat(va_logits, dim=0)
    va_targets_t = torch.cat(va_targets, dim=0)
    result = TrainResult(
        train_loss=float(sum(tr_losses) / max(1, len(tr_losses))),
        val_loss=float(sum(va_losses) / max(1, len(va_losses))),
        train_acc=accuracy_top1(tr_logits_t, tr_targets_t),
        val_acc=accuracy_top1(va_logits_t, va_targets_t),
        train_f1=f1_top1(tr_logits_t, tr_targets_t),
        val_f1=f1_top1(va_logits_t, va_targets_t),
        gates=last_gates,
    )
    return result

