from __future__ import annotations

import torch

from multiViewLearning.models import FourMultiCNN, MultiCNN, MultiViewResNet50Adaptive, SimpleCNN


def test_simplecnn_forward() -> None:
    model = SimpleCNN(in_channels=3, num_classes=3)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 3)


def test_multicnn_forward() -> None:
    model = MultiCNN(n_views=2, in_channels=3, num_classes=3)
    xs = [torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)]
    out = model(xs)
    assert out.shape == (2, 3)


def test_fourmulticnn_forward() -> None:
    model = FourMultiCNN(in_channels=3, num_classes=2)
    xs = [torch.randn(2, 3, 32, 32) for _ in range(4)]
    out = model(xs)
    assert out.shape == (2, 2)


def test_mv_resnet50_adaptive_forward() -> None:
    model = MultiViewResNet50Adaptive(n_views=2, in_channels=3, num_classes=2, gated=True)
    xs = [torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)]
    logits, gates = model(xs)
    assert logits.shape == (2, 2)
    assert gates is None or gates.shape[1] == 2

