from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml

from .datasets import DualViewDataset
from .models import MultiCNN
from .train import train_model


def _train(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("root", "data")
    views = cfg.get("views", ["rgb", "ir"])
    num_classes = int(cfg.get("num_classes", 2))
    batch_size = int(cfg.get("batch_size", 8))
    epochs = int(cfg.get("epochs", 1))

    train_ds = DualViewDataset(root=root, split="train", view_names=views)
    val_ds = DualViewDataset(root=root, split="val", view_names=views)
    model = MultiCNN(n_views=len(views), num_classes=num_classes)
    res = train_model(model, train_ds, val_ds, epochs=epochs, batch_size=batch_size)
    out = Path(cfg.get("out_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.txt").write_text(
        f"train_loss={res.train_loss}\nval_loss={res.val_loss}\ntrain_acc={res.train_acc}\nval_acc={res.val_acc}\ntrain_f1={res.train_f1}\nval_f1={res.val_f1}\n",
        encoding="utf-8",
    )
    print("Saved metrics to", out / "metrics.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvl", description="multiViewLearning CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_tr = sub.add_parser("train", help="Train from YAML config")
    p_tr.add_argument("--config", required=True)
    p_tr.set_defaults(func=_train)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

