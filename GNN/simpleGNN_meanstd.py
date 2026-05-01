import copy
import os
import random
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import numpy as np
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
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 20
THRESHOLD = 0.4

# Small validation-based search space. Keep this compact unless you want a longer run.
HIDDEN_DIM_OPTIONS = [64]
DROPOUT_OPTIONS = [0.3]
LR_OPTIONS = [1e-2]
WEIGHT_DECAY_OPTIONS = [1e-6]
THRESHOLD_OPTIONS = [0.55]

# Run multiple seeds for a more stable estimate.
SEEDS = [7, 13, 21]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        self.dropout = dropout

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


def summarize_metric_dicts(metrics_list: List[dict]) -> Dict[str, Tuple[float, float]]:
    summary = {}
    metric_names = metrics_list[0].keys()
    for name in metric_names:
        values = [metrics[name] for metrics in metrics_list]
        summary[name] = (mean(values), pstdev(values))
    return summary


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
        raise ValueError("No training graphs contained labeled nodes for supervision.")

    return total_loss / used_graphs


@torch.no_grad()
def collect_eval_outputs(
    model: nn.Module,
    graphs: List[Data],
    criterion: nn.Module,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()

    all_true = []
    all_probs = []
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

        all_true.append(data.y[data.supervision_mask].cpu())
        all_probs.append(probs[data.supervision_mask].cpu())

        total_loss += loss.item()
        used_graphs += 1

    if used_graphs == 0:
        raise ValueError("No evaluation graphs contained labeled nodes for supervision.")

    y_true = torch.cat(all_true, dim=0)
    y_prob = torch.cat(all_probs, dim=0)
    return total_loss / used_graphs, y_true, y_prob


def metrics_from_probs(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    threshold: float,
) -> dict:
    y_pred = (y_prob >= threshold).long()
    return compute_metrics(y_true, y_pred)


def find_best_threshold(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    thresholds: List[float],
) -> Tuple[float, dict]:
    best_threshold = thresholds[0]
    best_metrics = metrics_from_probs(y_true, y_prob, best_threshold)

    for threshold in thresholds[1:]:
        metrics = metrics_from_probs(y_true, y_prob, threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def build_model_and_training_objects(
    in_channels: int,
    num_classes: int,
    all_labels: torch.Tensor,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    model = SimpleGCN(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=num_classes,
        dropout=dropout,
    ).to(DEVICE)

    class_counts = torch.bincount(all_labels, minlength=num_classes).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    return model, criterion, optimizer


def train_and_select_model(
    train_graphs: List[Data],
    val_graphs: List[Data],
    in_channels: int,
    num_classes: int,
    all_labels: torch.Tensor,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    verbose: bool = True,
) -> Tuple[nn.Module, nn.Module, float, dict, float]:
    model, criterion, optimizer = build_model_and_training_objects(
        in_channels=in_channels,
        num_classes=num_classes,
        all_labels=all_labels,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_graphs, optimizer, criterion)
        val_loss, val_y_true, val_y_prob = collect_eval_outputs(model, val_graphs, criterion)
        current_threshold, current_metrics = find_best_threshold(
            val_y_true,
            val_y_prob,
            THRESHOLD_OPTIONS,
        )

        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Thr: {current_threshold:.2f} | "
                f"Val F1: {current_metrics['f1']:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, val_y_true, val_y_prob = collect_eval_outputs(model, val_graphs, criterion)
    best_threshold, best_val_metrics = find_best_threshold(val_y_true, val_y_prob, THRESHOLD_OPTIONS)

    return model, criterion, best_threshold, best_val_metrics, final_val_loss


def run_hyperparameter_search(
    train_graphs: List[Data],
    val_graphs: List[Data],
    in_channels: int,
    num_classes: int,
    all_labels: torch.Tensor,
    seed: int,
) -> Tuple[nn.Module, nn.Module, float, Dict[str, float], dict, float]:
    best_model = None
    best_criterion = None
    best_threshold = THRESHOLD
    best_config = None
    best_val_metrics = None
    best_val_loss = None
    best_val_f1 = -1.0

    search_space = list(
        product(
            HIDDEN_DIM_OPTIONS,
            DROPOUT_OPTIONS,
            LR_OPTIONS,
            WEIGHT_DECAY_OPTIONS,
        )
    )

    print(f"\nSeed {seed} | Running {len(search_space)} validation trials...")

    for trial_idx, (hidden_dim, dropout, lr, weight_decay) in enumerate(search_space, start=1):
        set_seed(seed)
        print(
            f"\nSeed {seed} | Trial {trial_idx:02d}/{len(search_space)} | "
            f"hidden_dim={hidden_dim}, dropout={dropout}, lr={lr}, weight_decay={weight_decay}"
        )

        model, criterion, threshold, val_metrics, val_loss = train_and_select_model(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            in_channels=in_channels,
            num_classes=num_classes,
            all_labels=all_labels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            verbose=False,
        )

        print(
            f"Trial result | Val Loss: {val_loss:.4f} | "
            f"Val Threshold: {threshold:.2f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model = copy.deepcopy(model)
            best_criterion = criterion
            best_threshold = threshold
            best_config = {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
            }
            best_val_metrics = val_metrics
            best_val_loss = val_loss

    return best_model, best_criterion, best_threshold, best_config, best_val_metrics, best_val_loss


def run_single_seed(
    train_graphs: List[Data],
    val_graphs: List[Data],
    test_graphs: List[Data],
    in_channels: int,
    num_classes: int,
    all_labels: torch.Tensor,
    seed: int,
) -> dict:
    set_seed(seed)

    best_model, best_criterion, best_threshold, best_config, best_val_metrics, best_val_loss = run_hyperparameter_search(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        in_channels=in_channels,
        num_classes=num_classes,
        all_labels=all_labels,
        seed=seed,
    )

    test_loss, test_y_true, test_y_prob = collect_eval_outputs(best_model, test_graphs, best_criterion)
    test_metrics = metrics_from_probs(test_y_true, test_y_prob, best_threshold)

    return {
        "seed": seed,
        "best_config": best_config,
        "best_threshold": best_threshold,
        "val_loss": best_val_loss,
        "val_metrics": best_val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }


# Main
def main():
    train_graphs = load_graphs_from_csv(TRAIN_SPLIT, GRAPH_ROOT)
    val_graphs = load_graphs_from_csv(VAL_SPLIT, GRAPH_ROOT)
    test_graphs = load_graphs_from_csv(TEST_SPLIT, GRAPH_ROOT)

    first_graph = train_graphs[0]
    in_channels = first_graph.x.size(1)

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

    print(f"Using device: {DEVICE}")
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs:   {len(val_graphs)}")
    print(f"Test graphs:  {len(test_graphs)}")
    print(f"Input dim:    {in_channels}")
    print(f"Num classes:  {num_classes}")
    print(f"Seeds:        {SEEDS}")
    print()

    run_results = []
    for seed in SEEDS:
        result = run_single_seed(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            test_graphs=test_graphs,
            in_channels=in_channels,
            num_classes=num_classes,
            all_labels=all_labels,
            seed=seed,
        )
        run_results.append(result)

        print(f"\nSeed {seed} summary")
        print(f"Best validation config: {result['best_config']}")
        print(f"Best validation threshold: {result['best_threshold']:.2f}")
        print(f"Validation loss: {result['val_loss']:.4f}")
        print(f"Validation F1:   {result['val_metrics']['f1']:.4f}")
        print(f"Test loss:       {result['test_loss']:.4f}")
        print(f"Test accuracy:   {result['test_metrics']['accuracy']:.4f}")
        print(f"Test precision:  {result['test_metrics']['precision']:.4f}")
        print(f"Test recall:     {result['test_metrics']['recall']:.4f}")
        print(f"Test F1:         {result['test_metrics']['f1']:.4f}")

    test_metric_summary = summarize_metric_dicts([result["test_metrics"] for result in run_results])
    test_losses = [result["test_loss"] for result in run_results]
    thresholds = [result["best_threshold"] for result in run_results]

    print("\nFinal Test Results Across Seeds")
    print(f"Threshold mean +/- std: {mean(thresholds):.4f} +/- {pstdev(thresholds):.4f}")
    print(f"Test Loss mean +/- std: {mean(test_losses):.4f} +/- {pstdev(test_losses):.4f}")
    print(
        f"Test Accuracy mean +/- std: {test_metric_summary['accuracy'][0]:.4f} +/- "
        f"{test_metric_summary['accuracy'][1]:.4f}"
    )
    print(
        f"Test Precision mean +/- std: {test_metric_summary['precision'][0]:.4f} +/- "
        f"{test_metric_summary['precision'][1]:.4f}"
    )
    print(
        f"Test Recall mean +/- std: {test_metric_summary['recall'][0]:.4f} +/- "
        f"{test_metric_summary['recall'][1]:.4f}"
    )
    print(
        f"Test F1 mean +/- std: {test_metric_summary['f1'][0]:.4f} +/- "
        f"{test_metric_summary['f1'][1]:.4f}"
    )


if __name__ == "__main__":
    main()
