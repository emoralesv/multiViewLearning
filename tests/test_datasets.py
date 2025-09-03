from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from multiViewLearning.datasets import DualViewDataset, make_weighted_sampler


def _make_dualview(tmp_path: Path, n: int = 6) -> Path:
    root = tmp_path / "data"
    (root / "train" / "rgb").mkdir(parents=True)
    (root / "train" / "ir").mkdir(parents=True)
    (root / "val" / "rgb").mkdir(parents=True)
    (root / "val" / "ir").mkdir(parents=True)
    rows = []
    for split in ["train", "val"]:
        for i in range(n):
            fn = f"{i}.png"
            Image.new("RGB", (32, 32), color=(i, i, i)).save(root / split / "rgb" / fn)
            Image.new("RGB", (32, 32), color=(i, i, i)).save(root / split / "ir" / fn)
            rows.append({"split": split, "filename": fn, "label": int(i % 2)})
        with open(root / split / "labels.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            for r in [r for r in rows if r["split"] == split]:
                writer.writerow({"filename": r["filename"], "label": r["label"]})
    return root


def test_dualview_dataset_loading(tmp_path: Path) -> None:
    root = _make_dualview(tmp_path)
    ds = DualViewDataset(root=root, split="train", view_names=("rgb", "ir"))
    sample = ds[0]
    assert len(sample.views) == 2
    assert sample.views[0].shape[0] == 3
    assert isinstance(sample.label, int)


def test_weighted_sampler(tmp_path: Path) -> None:
    root = _make_dualview(tmp_path)
    ds = DualViewDataset(root=root, split="train", view_names=("rgb", "ir"))
    labels = [ds.labels_map[fn] for fn in ds.files]
    sampler = make_weighted_sampler(labels)
    assert sampler is not None

