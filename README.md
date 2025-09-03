# multiViewLearning

Architectures and training utilities for synchronized multi-view learning.

- Datasets: `DualViewDataset` and utilities for multiple views
- Models: `SimpleCNN`, `MultiCNN`, `FourMultiCNN`, `MultiViewResNet50_adaptive` (gated)
- Training: `train_model(...)` returning losses/accs/f1s and optional gates
- CLI: `mvl train --config configs/mvl_baseline.yaml`

## Install

```
pip install -r requirements.txt
pip install -e .
```

## Quickstart

```python
from multiViewLearning.datasets import DualViewDataset
from multiViewLearning.models import MultiCNN

ds = DualViewDataset(root='data', split='train', view_names=('rgb','ir'))
model = MultiCNN(num_classes=2)
```

## CLI

```
mvl train --config configs/mvl_baseline.yaml
```

## Notes

- Uses PyTorch; runs on CPU or GPU
- Torchvision backbones optional; defaults to small custom CNN
- WeightedRandomSampler helper is included

