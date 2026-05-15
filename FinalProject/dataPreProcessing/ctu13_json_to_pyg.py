#!/usr/bin/env python3
"""
Convert CTU-13 stdlib graph-builder JSON/JSON.GZ files into PyTorch Geometric Data objects.

Expected input:
- Output directory from ctu13_graph_builder_stdlib.py
  with graph files under: <input-root>/graphs/<scenario>/*.json.gz (or .json)
- Optional metadata under: <input-root>/metadata/

This script requires:
- torch
- torch-geometric

It does NOT require pandas or numpy.

Example:
    python3 ctu13_json_to_pyg.py \
      --input-root ./ctu13_output \
      --output-dir ./ctu13_pyg \
      --verbose
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch_geometric.data import Data


DEFAULT_GRAPH_GLOBS = ("*.json.gz", "*.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CTU-13 JSON graph snapshots into PyTorch Geometric .pt files."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory produced by ctu13_graph_builder_stdlib.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where converted .pt files and metadata will be written.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Graph filename glob(s) relative to input-root/graphs. Default: *.json.gz and *.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files if present.",
    )
    parser.add_argument(
        "--save-manifest-only",
        action="store_true",
        help="Write the new metadata index but do not save .pt graph files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while converting.",
    )
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> Tuple[Path, Path]:
    graphs_dir = output_dir / "graphs"
    meta_dir = output_dir / "metadata"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir, meta_dir


def load_json(path: Path) -> Dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_graph_files(input_root: Path, patterns: Sequence[str]) -> List[Path]:
    graphs_root = input_root / "graphs"
    if not graphs_root.exists():
        raise FileNotFoundError(f"Input graphs directory not found: {graphs_root}")

    results: List[Path] = []
    seen = set()
    for pattern in patterns:
        for path in graphs_root.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                results.append(path)
    results.sort()
    return results


def as_float_matrix(value: Any, name: str) -> List[List[float]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    matrix: List[List[float]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError(f"{name} rows must be lists")
        matrix.append([float(x) for x in row])
    return matrix


def as_int_list(value: Any, name: str) -> List[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [int(x) for x in value]


def as_bool_list(value: Any, name: str) -> List[bool]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [bool(x) for x in value]


def as_edge_pairs(value: Any) -> List[List[int]]:
    if not isinstance(value, list):
        raise ValueError("edges must be a list")
    edges: List[List[int]] = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each edge must be a 2-item list")
        edges.append([int(pair[0]), int(pair[1])])
    return edges


def build_data_object(payload: Dict[str, Any]) -> Data:
    node_ips = payload.get("node_ips", [])
    if not isinstance(node_ips, list):
        raise ValueError("node_ips must be a list")

    x_list = as_float_matrix(payload.get("x", []), "x")
    y_list = as_int_list(payload.get("y", []), "y")
    labeled_mask_list = as_bool_list(payload.get("labeled_mask", []), "labeled_mask")
    edges_list = as_edge_pairs(payload.get("edges", []))
    edge_attr_list = as_float_matrix(payload.get("edge_attr", []), "edge_attr")

    num_nodes = len(node_ips)
    if len(x_list) != num_nodes:
        raise ValueError(f"x row count {len(x_list)} != node_ips count {num_nodes}")
    if len(y_list) != num_nodes:
        raise ValueError(f"y length {len(y_list)} != node count {num_nodes}")
    if len(labeled_mask_list) != num_nodes:
        raise ValueError(
            f"labeled_mask length {len(labeled_mask_list)} != node count {num_nodes}"
        )
    if len(edges_list) != len(edge_attr_list):
        raise ValueError(
            f"edges count {len(edges_list)} != edge_attr row count {len(edge_attr_list)}"
        )

    if x_list:
        x = torch.tensor(x_list, dtype=torch.float32)
    else:
        x = torch.empty((0, 0), dtype=torch.float32)

    if edges_list:
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    if edge_attr_list:
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    else:
        edge_attr = torch.empty((0, 0), dtype=torch.float32)

    y = torch.tensor(y_list, dtype=torch.long)
    labeled_mask = torch.tensor(labeled_mask_list, dtype=torch.bool)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        labeled_mask=labeled_mask,
    )

    # Preserve useful metadata as plain attributes.
    data.node_ips = node_ips
    data.node_feature_names = payload.get("node_feature_names", [])
    data.edge_feature_names = payload.get("edge_feature_names", [])
    data.scenario_name = payload.get("scenario_name", "")
    data.source_file = payload.get("source_file", "")
    data.window_start = payload.get("window_start", "")
    data.window_end = payload.get("window_end", "")
    data.num_nodes_original = num_nodes
    data.num_edges_original = len(edges_list)

    return data


class ManifestWriter:
    def __init__(self, path: Path):
        self.path = path
        self._fh = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=[
                "scenario_name",
                "source_json_path",
                "pt_path",
                "source_file",
                "window_start",
                "window_end",
                "num_nodes",
                "num_edges",
                "num_labeled_nodes",
                "num_positive_nodes",
            ],
        )
        self._writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)

    def close(self) -> None:
        self._fh.close()


def save_feature_metadata(input_root: Path, meta_dir: Path) -> None:
    src = input_root / "metadata" / "feature_names.json"
    dst = meta_dir / "feature_names.json"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def convert_one_graph(
    graph_path: Path,
    graphs_out_dir: Path,
    overwrite: bool,
    save_manifest_only: bool,
) -> Dict[str, Any]:
    payload = load_json(graph_path)
    data = build_data_object(payload)

    scenario_name = str(payload.get("scenario_name") or graph_path.parent.name)
    stem = graph_path.name
    if stem.endswith(".json.gz"):
        stem = stem[:-8]
    elif stem.endswith(".json"):
        stem = stem[:-5]

    scenario_out_dir = graphs_out_dir / scenario_name
    scenario_out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = scenario_out_dir / f"{stem}.pt"

    if not save_manifest_only and (overwrite or not pt_path.exists()):
        torch.save(data, pt_path)

    y_list = data.y.tolist()
    labeled_list = data.labeled_mask.tolist()
    num_labeled = sum(1 for keep in labeled_list if keep)
    num_positive = sum(1 for y, keep in zip(y_list, labeled_list) if keep and y == 1)

    return {
        "scenario_name": scenario_name,
        "source_json_path": str(graph_path),
        "pt_path": str(pt_path),
        "source_file": str(payload.get("source_file", "")),
        "window_start": str(payload.get("window_start", "")),
        "window_end": str(payload.get("window_end", "")),
        "num_nodes": int(data.num_nodes_original),
        "num_edges": int(data.num_edges_original),
        "num_labeled_nodes": int(num_labeled),
        "num_positive_nodes": int(num_positive),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    graphs_out_dir, meta_dir = ensure_dirs(output_dir)
    save_feature_metadata(input_root, meta_dir)

    patterns = tuple(args.glob) if args.glob else DEFAULT_GRAPH_GLOBS
    graph_files = find_graph_files(input_root, patterns)
    if not graph_files:
        raise FileNotFoundError(
            f"No graph JSON files found under {input_root / 'graphs'} for patterns {patterns}"
        )

    manifest = ManifestWriter(meta_dir / "graphs_index_pyg.csv")
    converted = 0
    failed = 0

    try:
        total = len(graph_files)
        for idx, graph_path in enumerate(graph_files, start=1):
            try:
                row = convert_one_graph(
                    graph_path=graph_path,
                    graphs_out_dir=graphs_out_dir,
                    overwrite=args.overwrite,
                    save_manifest_only=args.save_manifest_only,
                )
                manifest.write(row)
                converted += 1
                if args.verbose:
                    print(f"[{idx}/{total}] converted {graph_path}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"[ERROR] Failed to convert {graph_path}: {exc}", flush=True)
    finally:
        manifest.close()

    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "converted_graphs": converted,
        "failed_graphs": failed,
        "save_manifest_only": bool(args.save_manifest_only),
        "patterns": list(patterns),
    }
    (meta_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
