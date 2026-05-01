# CTU-13 PyG training setup

This guide explains how to:

1. create metadata and split files from the `.pt` graphs
2. inspect the splits
3. load the graphs into PyTorch Geometric
4. start training

## Required files

- `ctu13_pt_handoff_prep.py`
- `ctu13_pyg_loader.py`
- your `.pt` graph files

Each `.pt` graph should be a PyG `Data` object with at least:

- `x`
- `edge_index`
- `edge_attr`
- `y`
- `labeled_mask`

## Expected layout

```text
project/
  ctu13_pyg/
    graphs/
      CTU-Malware-Capture-Botnet-42/
        ...pt
      CTU-Malware-Capture-Botnet-43/
        ...pt
  metadata_out/
```

## 1) Create metadata and fixed splits

Use this if you want one train/val/test split.

```bash
python3 ctu13_pt_handoff_prep.py \
  --input-root ./ctu13_pyg \
  --output-dir ./metadata_out \
  --path-mode absolute \
  --write-fixed-split \
  --split-profile balanced \
  --verbose
```

This writes:

```text
metadata_out/
  metadata/
    graphs_index_pyg.csv
    feature_names.json
    split_train.csv
    split_val.csv
    split_test.csv
    splits.json
    handoff_summary.json
```

### Path mode

Use `--path-mode absolute` unless you specifically want relative paths.

If you use `--path-mode relative`, pass `--graph-root ./ctu13_pyg` to the loader.

## 2) Create grouped cross-validation folds

Use this if you want family-pair fold CSVs.

```bash
python3 ctu13_pt_handoff_prep.py \
  --input-root ./ctu13_pyg \
  --output-dir ./metadata_out \
  --path-mode absolute \
  --write-family-pair-folds \
  --verbose
```

This creates fold folders like:

```text
metadata_out/folds_family_pair/fold_001__val-Virut__test-Rbot/
  train.csv
  val.csv
  test.csv
  fold.json
```

## 3) Inspect the data before training

### Fixed split

```bash
python3 ctu13_pyg_loader.py \
  --metadata-dir ./metadata_out/metadata \
  --print-summary \
  --peek-batch
```

### One CV fold

```bash
python3 ctu13_pyg_loader.py \
  --fold-dir ./metadata_out/folds_family_pair/fold_001__val-Virut__test-Rbot \
  --print-summary \
  --peek-batch
```

If your manifests use relative graph paths, add:

```bash
--graph-root ./ctu13_pyg
```

## 4) Load the fixed split in Python

```python
from ctu13_pyg_loader import build_fixed_split_loaders

train_loader, val_loader, test_loader = build_fixed_split_loaders(
    metadata_dir="./metadata_out/metadata",
    batch_size=8,
)
```

If your CSVs use relative graph paths:

```python
train_loader, val_loader, test_loader = build_fixed_split_loaders(
    metadata_dir="./metadata_out/metadata",
    graph_root="./ctu13_pyg",
    batch_size=8,
)
```

## 5) Load one grouped-CV fold in Python

```python
from ctu13_pyg_loader import build_fold_loaders

train_loader, val_loader, test_loader = build_fold_loaders(
    fold_dir="./metadata_out/folds_family_pair/fold_001__val-Virut__test-Rbot",
    batch_size=8,
)
```

## 6) Start training

Minimal GraphSAGE example:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from ctu13_pyg_loader import build_fixed_split_loaders


device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, test_loader = build_fixed_split_loaders(
    metadata_dir="./metadata_out/metadata",
    batch_size=8,
)

sample_batch = next(iter(train_loader))
in_dim = sample_batch.x.size(-1)


class GraphSAGENodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        return self.lin(x)


model = GraphSAGENodeClassifier(in_dim=in_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


def run_epoch(loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_nodes = 0
    total_correct = 0

    for data in loader:
        data = data.to(device)
        mask = data.labeled_mask & (data.y >= 0)
        if int(mask.sum()) == 0:
            continue

        if train:
            optimizer.zero_grad()

        logits = model(data)
        loss = criterion(logits[mask], data.y[mask])

        if train:
            loss.backward()
            optimizer.step()

        preds = logits[mask].argmax(dim=-1)
        total_correct += int((preds == data.y[mask]).sum().item())
        total_nodes += int(mask.sum().item())
        total_loss += float(loss.item()) * int(mask.sum().item())

    avg_loss = total_loss / max(total_nodes, 1)
    acc = total_correct / max(total_nodes, 1)
    return avg_loss, acc


for epoch in range(1, 21):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    print(
        f"epoch={epoch:03d} "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )
```

## Important training detail

Always train on labeled nodes only:

```python
mask = data.labeled_mask & (data.y >= 0)
loss = criterion(logits[mask], data.y[mask])
```

## Minimal workflow summary

### Fixed split

```bash
python3 ctu13_pt_handoff_prep.py \
  --input-root ./ctu13_pyg \
  --output-dir ./metadata_out \
  --path-mode absolute \
  --write-fixed-split \
  --split-profile balanced \
  --verbose

python3 ctu13_pyg_loader.py \
  --metadata-dir ./metadata_out/metadata \
  --print-summary \
  --peek-batch
```

Then train with `build_fixed_split_loaders(...)`.

### Grouped cross-validation

```bash
python3 ctu13_pt_handoff_prep.py \
  --input-root ./ctu13_pyg \
  --output-dir ./metadata_out \
  --path-mode absolute \
  --write-family-pair-folds \
  --verbose

python3 ctu13_pyg_loader.py \
  --fold-dir ./metadata_out/folds_family_pair/fold_001__val-Virut__test-Rbot \
  --print-summary \
  --peek-batch
```

Then train with `build_fold_loaders(...)`.
