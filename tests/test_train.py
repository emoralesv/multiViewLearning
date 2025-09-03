from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from multiViewLearning.datasets import DualViewDataset
from multiViewLearning.models import MultiCNN
from multiViewLearning.train import train_model


def _make_data(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "train" / "rgb").mkdir(parents=True)
    (root / "train" / "ir").mkdir(parents=True)
    (root / "val" / "rgb").mkdir(parents=True)
    (root / "val" / "ir").mkdir(parents=True)
    for split in ["train", "val"]:
        rows = []
        for i in range(8):
            fn = f"{i}.png"
            Image.new("RGB", (32, 32), color=(i, i, i)).save(root / split / "rgb" / fn)
            Image.new("RGB", (32, 32), color=(i, i, i)).save(root / split / "ir" / fn)
            rows.append({"filename": fn, "label": int(i % 2)})
        with open(root / split / "labels.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows(rows)
    return root


def test_train_model(tmp_path: Path) -> None:
    root = _make_data(tmp_path)
    train_ds = DualViewDataset(root=root, split="train", view_names=("rgb", "ir"))
    val_ds = DualViewDataset(root=root, split="val", view_names=("rgb", "ir"))
    model = MultiCNN(n_views=2, num_classes=2)
    res = train_model(model, train_ds, val_ds, epochs=1, batch_size=4)
    assert 0.0 <= res.train_acc <= 1.0
    assert 0.0 <= res.val_acc <= 1.0

