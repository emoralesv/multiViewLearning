from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


Transform = Callable[[Image.Image], torch.Tensor]


@dataclass
class DualViewSample:
    views: List[torch.Tensor]
    label: int


class DualViewDataset(Dataset[DualViewSample]):
    """Synchronized two-view dataset from directory structure.

    Expects structure:
      root/split/<view_name>/*.jpg and a CSV labels file: root/split/labels.csv with columns
      filename,label where filename is the basename (without directory) shared across views.

    Args:
        root: Root directory path.
        split: Split name (e.g., 'train', 'val').
        view_names: Tuple of view directory names.
        transform_per_view: Optional transforms per view (len must match views) returning tensor.
        joint_transform: Optional transform applied to list of PIL images before per-view transforms.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        view_names: Sequence[str] = ("rgb", "ir"),
        transform_per_view: Sequence[Transform] | None = None,
        joint_transform: Callable[[List[Image.Image]], List[Image.Image]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.view_names = list(view_names)
        self.split_dir = self.root / split
        self.joint_transform = joint_transform
        if transform_per_view is None:
            # default: convert to CHW tensor in [0,1]
            def to_tensor(img: Image.Image) -> torch.Tensor:
                arr = np.array(img, dtype=np.float32) / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                chw = torch.from_numpy(arr).permute(2, 0, 1)
                return chw

            self.transforms = [to_tensor for _ in self.view_names]
        else:
            assert len(transform_per_view) == len(self.view_names)
            self.transforms = list(transform_per_view)

        # load labels
        labels_csv = self.split_dir / "labels.csv"
        if not labels_csv.exists():
            raise FileNotFoundError(f"labels.csv not found at {labels_csv}")
        with open(labels_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.labels_map = {row["filename"]: int(row["label"]) for row in rows}

        # collect filenames present in all views
        views_files = []
        for vn in self.view_names:
            vdir = self.split_dir / vn
            views_files.append({p.name: p for p in sorted(vdir.glob("*")) if p.is_file()})
        common = set.intersection(*[set(d.keys()) for d in views_files]) if views_files else set()
        self.files = sorted([fn for fn in common if fn in self.labels_map])
        self.view_dirs = [self.split_dir / vn for vn in self.view_names]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> DualViewSample:
        fn = self.files[idx]
        imgs = [Image.open(vd / fn).convert("RGB") for vd in self.view_dirs]
        if self.joint_transform:
            imgs = self.joint_transform(imgs)
        tensors = [t(img) for t, img in zip(self.transforms, imgs)]
        label = self.labels_map[fn]
        return DualViewSample(views=tensors, label=label)


def make_weighted_sampler(labels: Iterable[int]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for class-balanced sampling.

    Args:
        labels: Iterable of labels.

    Returns:
        WeightedRandomSampler instance.
    """
    labels_list = list(labels)
    class_counts = {}
    for y in labels_list:
        class_counts[y] = class_counts.get(y, 0) + 1
    weights = [1.0 / class_counts[y] for y in labels_list]
    return WeightedRandomSampler(weights, num_samples=len(labels_list), replacement=True)

