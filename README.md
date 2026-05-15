# GNN Botnet Detection Scripts

This folder contains four Python scripts for training and evaluating Graph Neural Network (GNN) models for botnet detection using PyTorch and PyTorch Geometric.

The scripts are designed to use graph snapshots referenced by metadata CSV files. Each graph is expected to be stored as a PyTorch Geometric `Data` object.

## Included Scripts

| Script | Description |
|---|---|
| `baselineGCN.py` | Standard 2-layer GCN baseline. Uses node features and graph connectivity only. Does **not** use thresholding. Predictions are made with `argmax(logits)`. |
| `thresholdGCN.py` | Same 2-layer GCN architecture as the baseline, but uses probability thresholding on the malicious class probability. |
| `edge-awareGNN.py` | Edge-aware attention-style GNN that uses node features, graph connectivity, and edge attributes. |
| `deepGNN.py` | 3-layer GCN model used to test whether a larger graph receptive field improves performance. |

## Expected Project Structure

The scripts assume the following relative paths from the directory where the script is run:

```text
project_root/
├── baselineGCN.py
├── thresholdGCN.py
├── edge-awareGNN.py
├── deepGNN.py
├── packagedData/
│   ├── graphs/
│   │   └── ...
│   └── metadata_out/
│       └── metadata/
│           ├── split_train.csv
│           ├── split_val.csv
│           └── split_test.csv
```

The CSV files should contain a column named:

```text
graph_relpath
```

Each value in `graph_relpath` should point to a graph file relative to:

```text
./packagedData/graphs
```

## Required Graph Fields

Each graph file should be a PyTorch Geometric `Data` object.

For the baseline, thresholded, and deeper GCN scripts, each graph must contain:

```python
data.x
data.edge_index
data.y
```

For the edge-aware GNN script, each graph must also contain:

```python
data.edge_attr
```

If available, the scripts use:

```python
data.labeled_mask
```

to select valid supervised nodes. If `labeled_mask` is not available, all nodes are treated as supervised.

## Label Assumptions

The scripts assume binary node classification:

```text
0 = benign / normal
1 = botnet / malicious
```

Nodes with invalid or ignored labels should be excluded using `labeled_mask`.

## Dependencies

Install the required packages in your Python environment:

```bash
python -m pip install pandas scikit-learn
python -m pip install torch torchvision torchaudio
python -m pip install torch-geometric
```

Depending on your machine and CUDA setup, PyTorch and PyTorch Geometric may require platform-specific installation commands.

For GPU support, verify that PyTorch can see CUDA:

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

If `torch.cuda.is_available()` returns `False` and `torch.version.cuda` is `None`, then you likely installed the CPU-only PyTorch build.

## Running the Scripts

Run any script from the project root:

```bash
python baseline_gcn.py
python thresholded_gcn.py
python edge_aware_gnn.py
python deeper_gcn.py
```

Each script will:

1. Load graphs from `split_train.csv`, `split_val.csv`, and `split_test.csv`
2. Normalize node features
3. Train on the training graph split
4. Track best validation F1
5. Restore the best validation model
6. Evaluate final performance on the test graph split

The edge-aware script additionally normalizes edge features and passes `edge_attr` into the model.

## Main Hyperparameters

Each script has a configuration and hyperparameters section near the top:

```python
HIDDEN_DIM = 32
DROPOUT = 0.3
LR = 1e-2
WEIGHT_DECAY = 1e-6
EPOCHS = 100
PATIENCE = 50
```

For threshold-based scripts, the default threshold is:

```python
THRESHOLD = 0.60
```

The thresholded scripts also perform a validation sweep to select the best threshold before final test evaluation.

## Model Notes

### Baseline GCN

The baseline model uses:

```text
24 -> 32 -> 32 -> 2
```

assuming 24 node input features and 2 output classes.

It uses standard GCN message passing through `GCNConv` and predicts labels using:

```python
preds = logits.argmax(dim=1)
```

### Thresholded GCN

The thresholded model uses the same GCN architecture as the baseline but changes the prediction rule.

Instead of `argmax`, it computes:

```python
probs = torch.softmax(logits, dim=1)[:, 1]
preds = (probs >= threshold).long()
```

This allows precision and recall to be adjusted through a probability threshold.

### Edge-Aware GNN

The edge-aware model uses a custom message-passing layer. It incorporates:

- source node embeddings
- target node embeddings
- edge feature embeddings

into the attention/message computation.

This model requires `edge_attr`.

### Deeper GCN

The deeper GCN adds a third graph convolution layer:

```text
24 -> 32 -> 32 -> 32 -> 2
```

This increases the approximate receptive field from a two-hop neighborhood to a three-hop neighborhood.

## Output Metrics

Each script reports:

- Loss
- Accuracy
- Precision
- Recall
- F1 score

F1 score is generally the most important metric for this project because botnet detection is an imbalanced binary classification task.

## Notes on Reproducibility

These scripts do not include explicit seed-setting code by default. Results may vary between runs due to random initialization and training dynamics.

For formal experiments, run each configuration across multiple random seeds and report mean plus standard deviation.

## Generated Notice

This README was generated by ChatGPT.
