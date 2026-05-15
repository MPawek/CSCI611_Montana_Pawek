import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# Hyperparameters and configurations
TRAIN_SPLIT = "./packagedData/metadata_out/metadata/split_train.csv"
VAL_SPLIT = "./packagedData/metadata_out/metadata/split_val.csv"
TEST_SPLIT = "./packagedData/metadata_out/metadata/split_test.csv"

GRAPH_ROOT = "./packagedData/graphs"

HIDDEN_DIM = 32
DROPOUT = 0.3
LR = 1e-2
WEIGHT_DECAY = 1e-6
EPOCHS = 100
PATIENCE = 50

# Used only by thresholded versions.
THRESHOLD = 0.60

# Make sure CUDA is enabled
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


HEADS = 4

# CSV Loading
# Borrowed largely from examples on GitHub
def load_metadata(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    if "graph_relpath" not in df.columns:
        raise ValueError("CSV is missing required column: graph_relpath")

    return df


def resolve_graph_path(graph_root: str, graph_relpath: str) -> str:
    relpath = Path(str(graph_relpath).replace("\\", os.sep))
    return str(Path(graph_root) / relpath)


def normalize_node_features(data: Data) -> Data:
    data.x = data.x.float()

    x_mean = data.x.mean(dim=0, keepdim=True)
    x_std = data.x.std(dim=0, keepdim=True).clamp_min(1e-6)
    data.x = (data.x - x_mean) / x_std

    return data


def normalize_node_and_edge_features(data: Data) -> Data:
    data = normalize_node_features(data)

    data.edge_attr = data.edge_attr.float()
    e_mean = data.edge_attr.mean(dim=0, keepdim=True)
    e_std = data.edge_attr.std(dim=0, keepdim=True).clamp_min(1e-6)
    data.edge_attr = (data.edge_attr - e_mean) / e_std

    return data


def ensure_label_vector(data: Data) -> Data:
    if data.y.dim() > 1:
        data.y = data.y.view(-1)
    data.y = data.y.long()
    return data


def add_supervision_mask(data: Data) -> Data:
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
    require_edge_attr: bool = False,
) -> List[Data]:
    df = load_metadata(csv_path)

    graphs = []
    skipped = 0

    required_fields = ["x", "edge_index", "y"]
    if require_edge_attr:
        required_fields.append("edge_attr")

    for idx, row in df.iterrows():
        graph_path = resolve_graph_path(graph_root, row["graph_relpath"])

        if not os.path.exists(graph_path):
            print(f"Skipping missing graph: {graph_path}")
            skipped += 1
            continue

        data = torch.load(graph_path, map_location="cpu", weights_only=False)

        if not all(hasattr(data, field) for field in required_fields):
            print(f"Skipping graph with missing fields: {graph_path}")
            skipped += 1
            continue

        if require_edge_attr:
            data = normalize_node_and_edge_features(data)
        else:
            data = normalize_node_features(data)

        data = ensure_label_vector(data)
        data = add_supervision_mask(data)

        data.graph_path = graph_path
        data.csv_row_index = idx

        graphs.append(data)

    if not graphs:
        raise ValueError(f"No usable graphs were loaded from: {csv_path}")

    print(f"Loaded {len(graphs)} graphs from {csv_path}. Skipped {skipped}.")
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


def infer_num_classes(graphs: List[Data]) -> int:
    all_labels = []

    for graph in graphs:
        labels = graph.y[graph.supervision_mask]
        if labels.numel() > 0:
            all_labels.append(labels)

    if not all_labels:
        raise ValueError("Could not infer classes: no supervised labels found.")

    all_labels = torch.cat(all_labels, dim=0)
    unique_classes = torch.unique(all_labels)

    if unique_classes.numel() < 2:
        raise ValueError(
            f"Training data contains fewer than 2 classes: {unique_classes.tolist()}"
        )

    return int(unique_classes.numel())


def compute_class_weights(graphs: List[Data], num_classes: int) -> torch.Tensor:
    all_labels = []

    for graph in graphs:
        labels = graph.y[graph.supervision_mask]
        if labels.numel() > 0:
            all_labels.append(labels)

    all_labels = torch.cat(all_labels, dim=0)
    class_counts = torch.bincount(all_labels, minlength=num_classes).float()

    class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
    class_weights = class_weights / class_weights.sum() * num_classes

    return class_weights


def print_split_stats(name: str, graphs: List[Data]) -> None:
    all_labels = []

    for graph in graphs:
        labels = graph.y[graph.supervision_mask]
        if labels.numel() > 0:
            all_labels.append(labels.cpu())

    y = torch.cat(all_labels, dim=0)
    unique, counts = torch.unique(y, return_counts=True)

    print(f"{name} label distribution:")
    total = int(counts.sum().item())
    for cls, count in zip(unique.tolist(), counts.tolist()):
        print(f"  class {cls}: {count} ({count / total:.4f})")
    print()

# Edge-Aware Message Passing
class EdgeAwareGATLayer(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.node_proj = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.edge_proj = nn.Linear(edge_dim, heads * out_channels, bias=False)

        self.attn = nn.Parameter(torch.Tensor(heads, 3 * out_channels))
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.attn)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x_proj = self.node_proj(x).view(-1, self.heads, self.out_channels)
        edge_proj = self.edge_proj(edge_attr).view(-1, self.heads, self.out_channels)

        out = self.propagate(
            edge_index=edge_index,
            x=x_proj,
            edge_attr=edge_proj,
        )

        out = out.reshape(-1, self.heads * self.out_channels)
        out = out + self.bias
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr,
        size_i,
    ) -> torch.Tensor:
        attn_input = torch.cat([x_i, x_j, edge_attr], dim=-1)

        alpha = (attn_input * self.attn).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_j + edge_attr
        return msg * alpha.unsqueeze(-1)


# Edge-Aware GNN
class EdgeAwareGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_dim: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Added edge dimensions to each layer
        self.layer1 = EdgeAwareGATLayer(
            in_channels=in_channels,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
        )

        self.layer2 = EdgeAwareGATLayer(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
        )

        self.classifier = nn.Linear(hidden_dim * heads, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layer2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.classifier(x)
        return logits


# Training/Eval
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

        labels = data.y[data.supervision_mask]
        if labels.numel() == 0:
            continue

        optimizer.zero_grad()

        logits = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(logits[data.supervision_mask], data.y[data.supervision_mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        used_graphs += 1

    if used_graphs == 0:
        raise ValueError("No usable training graphs were found.")

    return total_loss / used_graphs


@torch.no_grad()
def evaluate_on_graphs(
    model: nn.Module,
    graphs: List[Data],
    criterion: nn.Module,
    threshold: float = THRESHOLD,
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

        logits = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(logits[data.supervision_mask], data.y[data.supervision_mask])

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long()

        all_true.append(data.y[data.supervision_mask].cpu())
        all_pred.append(preds[data.supervision_mask].cpu())

        total_loss += loss.item()
        used_graphs += 1

    if used_graphs == 0:
        raise ValueError("No usable evaluation graphs were found.")

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    metrics = compute_metrics(y_true, y_pred)

    return total_loss / used_graphs, metrics


@torch.no_grad()
def find_best_threshold(
    model: nn.Module,
    graphs: List[Data],
) -> Tuple[float, float]:
    model.eval()

    all_true = []
    all_probs = []

    for data in graphs:
        data = data.to(DEVICE)

        labels = data.y[data.supervision_mask]
        if labels.numel() == 0:
            continue

        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_true.append(data.y[data.supervision_mask].cpu())
        all_probs.append(probs[data.supervision_mask].cpu())

    y_true = torch.cat(all_true).numpy()
    y_prob = torch.cat(all_probs).numpy()

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in [i / 100 for i in range(10, 91, 5)]:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


# Main
def main():
    train_graphs = load_graphs_from_csv(TRAIN_SPLIT, GRAPH_ROOT, require_edge_attr=True)
    val_graphs = load_graphs_from_csv(VAL_SPLIT, GRAPH_ROOT, require_edge_attr=True)
    test_graphs = load_graphs_from_csv(TEST_SPLIT, GRAPH_ROOT, require_edge_attr=True)

    # Debug prints
    print_split_stats("Train", train_graphs)
    print_split_stats("Validation", val_graphs)
    print_split_stats("Test", test_graphs)

    in_channels = train_graphs[0].x.size(1)
    edge_dim = train_graphs[0].edge_attr.size(1)
    num_classes = infer_num_classes(train_graphs)

    model = EdgeAwareGNN(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
        out_channels=num_classes,
        heads=HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    class_weights = compute_class_weights(train_graphs, num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0

    print(f"Using device: {DEVICE}")
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs:   {len(val_graphs)}")
    print(f"Test graphs:  {len(test_graphs)}")
    print(f"Input dim:    {in_channels}")
    print(f"Edge dim:     {edge_dim}")
    print(f"Num classes:  {num_classes}")
    print(f"Heads:        {HEADS}")
    print(f"Default threshold: {THRESHOLD}")
    print()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_graphs, optimizer, criterion)
        val_loss, val_metrics = evaluate_on_graphs(model, val_graphs, criterion, threshold=THRESHOLD)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val Prec: {val_metrics['precision']:.4f} | "
                f"Val Rec: {val_metrics['recall']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_threshold, threshold_val_f1 = find_best_threshold(model, val_graphs)

    print(f"Best validation F1 during training: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Best threshold from validation sweep: {best_threshold:.2f}")
    print(f"Validation F1 at best threshold: {threshold_val_f1:.4f}")

    test_loss, test_metrics = evaluate_on_graphs(
        model,
        test_graphs,
        criterion,
        threshold=best_threshold,
    )

    print("\nFinal Test Results")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
