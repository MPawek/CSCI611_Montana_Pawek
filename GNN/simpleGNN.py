import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# Configurations and Hyperparameters
TRAIN_SPLIT = "./packagedData/metadata_out/metadata/split_train.csv"
VAL_SPLIT = "./packagedData/metadata_out/metadata/split_val.csv"
TEST_SPLIT = "./packagedData/metadata_out/metadata/split_test.csv"

GRAPH_ROOT = "./packagedData/graphs"


HIDDEN_DIM = 32
DROPOUT = 0.3
LR = 1e-2
WEIGHT_DECAY = 1e-6
EPOCHS = 100
PATIENCE = 20
THRESHOLD = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initial GNN model with no added features
class SimpleGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Dropout for regularization
        self.dropout = dropout

        # Conv layers similar to a CNN
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.classifier(x)
        return logits


# CSV Loading
def load_metadata(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["graph_relpath"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV is missing required column: {col}")

    return df


def resolve_graph_path(graph_root: str, graph_relpath: str) -> str:
    """
    Converts a CSV graph_relpath like:
      CTU-Malware-Capture-Botnet-42\\CTU-Malware-Capture-Botnet-42__20110810_094500.pt

    into a valid local path under GRAPH_ROOT.
    """
    relpath = Path(graph_relpath.replace("\\", os.sep))
    full_path = Path(graph_root) / relpath
    return str(full_path)


# Data Preprocessing
def normalize_features(data: Data) -> Data:
    data.x = data.x.float()

    x_mean = data.x.mean(dim=0, keepdim=True)
    x_std = data.x.std(dim=0, keepdim=True).clamp_min(1e-6)
    data.x = (data.x - x_mean) / x_std

    return data


def ensure_label_vector(data: Data) -> Data:
    if data.y.dim() > 1:
        data.y = data.y.view(-1)
    data.y = data.y.long()
    return data

def add_supervision_mask(data: Data) -> Data:
    """
    Use labeled_mask if it exists.
    Otherwise supervise on all nodes.
    """
    if hasattr(data, "labeled_mask"):
        data.supervision_mask = data.labeled_mask.bool()
    else:
        data.supervision_mask = torch.ones(data.num_nodes, dtype=torch.bool)

    if int(data.supervision_mask.sum()) == 0:
        raise ValueError("Graph has no labeled nodes available for supervision.")

    return data

def load_graphs_from_csv(
    csv_path: str,
    graph_root: str,
) -> List[Data]:
    df = load_metadata(csv_path)

    graphs = []
    skipped = 0

    for idx, row in df.iterrows():
        graph_path = resolve_graph_path(graph_root, row["graph_relpath"])

        if not os.path.exists(graph_path):
            print(f"Skipping missing graph: {graph_path}")
            skipped += 1
            continue

        data = torch.load(graph_path, map_location="cpu", weights_only=False)

        required_fields = ["x", "edge_index", "y"]
        if not all(hasattr(data, field) for field in required_fields):
            print(f"Skipping graph with missing fields: {graph_path}")
            skipped += 1
            continue

        data = normalize_features(data)
        data = ensure_label_vector(data)
        data = add_supervision_mask(data)

        # Keep some metadata for debugging
        data.graph_path = graph_path
        data.csv_row_index = idx

        graphs.append(data)

    if not graphs:
        raise ValueError("No usable graphs were loaded from the training CSV.")

    print(f"Loaded {len(graphs)} graphs from training CSV. Skipped {skipped}.")
    return graphs



# Metrics
def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "precision": precision_score(y_true_np, y_pred_np, average="binary", zero_division=0),
        "recall": recall_score(y_true_np, y_pred_np, average="binary", zero_division=0),
        "f1": f1_score(y_true_np, y_pred_np, average="binary", zero_division=0),
    }


# Training and Evaluation
def train_one_epoch(
    model: nn.Module,
    graphs: List[Data],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    used_graphs = 0

    for data in graphs:
        data = data.to(DEVICE)

        # Skip graphs whose labeled nodes contain no classes
        labels = data.y[data.supervision_mask]
        if labels.numel() == 0:
            continue

        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.supervision_mask], data.y[data.supervision_mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        used_graphs += 1

    if used_graphs == 0:
        raise ValueError("No training graphs contained at least two supervised classes.")

    return total_loss / used_graphs


@torch.no_grad()
def evaluate_on_graphs(
    model: nn.Module,
    graphs: List[Data],
    criterion: nn.Module,
) -> Tuple[float, dict]:
    model.eval()

    all_true = []
    all_pred = []
    total_loss = 0.0
    used_graphs = 0

    for data in graphs:
        data = data.to(DEVICE)

        labels = data.y[data.supervision_mask]
        if labels.numel() == 0:
            continue

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.supervision_mask], data.y[data.supervision_mask])

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= THRESHOLD).long()

        all_true.append(data.y[data.supervision_mask].cpu())
        all_pred.append(preds[data.supervision_mask].cpu())

        total_loss += loss.item()
        used_graphs += 1

    if used_graphs == 0:
        raise ValueError("No evaluation graphs contained at least two supervised classes.")

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    metrics = compute_metrics(y_true, y_pred)

    return total_loss / used_graphs, metrics


# Main
def main():
    train_graphs = load_graphs_from_csv(TRAIN_SPLIT, GRAPH_ROOT)
    val_graphs = load_graphs_from_csv(VAL_SPLIT, GRAPH_ROOT)
    test_graphs = load_graphs_from_csv(TEST_SPLIT, GRAPH_ROOT)

    # Infer dimensions from first graph
    first_graph = train_graphs[0]
    in_channels = first_graph.x.size(1)

    # Infer classes across all training graphs
    all_labels = []
    for g in train_graphs:
        all_labels.append(g.y[g.supervision_mask])
    all_labels = torch.cat(all_labels, dim=0)

    print("Unique training labels:", torch.unique(all_labels).tolist())

    unique_classes = torch.unique(all_labels)
    num_classes = int(unique_classes.numel())

    if num_classes < 2:
        raise ValueError(
            f"All graphs together still contain fewer than 2 classes: {unique_classes.tolist()}"
        )

    model = SimpleGCN(
        in_channels=in_channels,
        hidden_dim=HIDDEN_DIM,
        out_channels=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    class_counts = torch.bincount(all_labels, minlength=num_classes).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    print(f"Using device: {DEVICE}")
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs:   {len(val_graphs)}")
    print(f"Test graphs:  {len(test_graphs)}")
    print(f"Input dim:    {in_channels}")
    print(f"Num classes:  {num_classes}")
    print()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_graphs, optimizer, criterion)
        val_loss, val_metrics = evaluate_on_graphs(model, val_graphs, criterion)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Eval Loss: {val_loss:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | "
                f"Prec: {val_metrics['precision']:.4f} | "
                f"Rec: {val_metrics['recall']:.4f} | "
                f"F1: {val_metrics['f1']:.4f}"
            )


        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = evaluate_on_graphs(model, test_graphs, criterion)

    print("\nFinal Test Results")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()