from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, feat_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        z = self.fc(h)
        z = F.relu(z, inplace=True)
        logits = self.classifier(z)
        return logits


class _Tower(nn.Module):
    def __init__(self, in_channels: int = 3, feat_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return F.relu(self.fc(h), inplace=True)


class MultiCNN(nn.Module):
    def __init__(self, n_views: int = 2, in_channels: int = 3, num_classes: int = 2, feat_dim: int = 128) -> None:
        super().__init__()
        self.n_views = n_views
        self.towers = nn.ModuleList([_Tower(in_channels, feat_dim) for _ in range(n_views)])
        self.classifier = nn.Linear(feat_dim * n_views, num_classes)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        feats = [tower(x) for tower, x in zip(self.towers, xs)]
        z = torch.cat(feats, dim=1)
        return self.classifier(z)


class FourMultiCNN(MultiCNN):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, feat_dim: int = 128) -> None:
        super().__init__(n_views=4, in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim)


class MultiViewResNet50Adaptive(nn.Module):
    """Adaptive gated fusion over per-view towers (simple CNN towers by default).

    This is a lightweight stand-in for a ResNet-based multi-view model.
    """

    def __init__(self, n_views: int = 2, in_channels: int = 3, num_classes: int = 2, feat_dim: int = 256, gated: bool = True) -> None:
        super().__init__()
        self.n_views = n_views
        self.gated = gated
        self.towers = nn.ModuleList([_Tower(in_channels, feat_dim) for _ in range(n_views)])
        if gated:
            self.gates = nn.ModuleList([nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid()) for _ in range(n_views)])
        else:
            self.gates = None
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, xs: List[torch.Tensor]):
        feats = [tower(x) for tower, x in zip(self.towers, xs)]
        if self.gated and self.gates is not None:
            weights = [gate(f) for gate, f in zip(self.gates, feats)]
            # weighted sum of features
            z = sum(w * f for w, f in zip(weights, feats)) / len(feats)
            gate_vals = torch.cat(weights, dim=1)
        else:
            z = sum(feats) / len(feats)
            gate_vals = None
        logits = self.classifier(z)
        return logits, gate_vals

