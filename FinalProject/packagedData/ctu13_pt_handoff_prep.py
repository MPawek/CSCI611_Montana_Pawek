#!/usr/bin/env python3
"""
Prepare metadata manifests and training split files directly from CTU-13 PyTorch Geometric
`.pt` graph files.

This script scans `.pt` graphs, extracts metadata from the graph objects themselves,
creates a clean master manifest, and optionally writes:

- fixed train/val/test split CSVs
- family-pair grouped cross-validation fold CSVs

It requires:
- torch
- torch-geometric (only so torch.load can deserialize PyG Data objects reliably)

It does NOT require:
- pandas
- numpy

Example:
    python3 ctu13_pt_handoff_prep.py \
      --input-root ./ctu13_pyg \
      --output-dir ./ctu13_handoff \
      --write-fixed-split \
      --split-profile strict \
      --write-family-pair-folds \
      --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data  # noqa: F401  # ensures class is importable for torch.load

SCENARIO_FAMILY_MAP: Dict[str, str] = {
    "CTU-Malware-Capture-Botnet-42": "Neris",
    "CTU-Malware-Capture-Botnet-43": "Neris",
    "CTU-Malware-Capture-Botnet-44": "Rbot",
    "CTU-Malware-Capture-Botnet-45": "Rbot",
    "CTU-Malware-Capture-Botnet-46": "Virut",
    "CTU-Malware-Capture-Botnet-47": "DonBot",
    "CTU-Malware-Capture-Botnet-48": "Sogou",
    "CTU-Malware-Capture-Botnet-49": "Murlo",
    "CTU-Malware-Capture-Botnet-50": "Neris",
    "CTU-Malware-Capture-Botnet-51": "Rbot",
    "CTU-Malware-Capture-Botnet-52": "Rbot",
    "CTU-Malware-Capture-Botnet-53": "NSIS.ay",
    "CTU-Malware-Capture-Botnet-54": "Virut",
}

# Stricter unseen-family split.
STRICT_SPLIT = {
    "train": [
        "CTU-Malware-Capture-Botnet-42",
        "CTU-Malware-Capture-Botnet-43",
        "CTU-Malware-Capture-Botnet-47",
        "CTU-Malware-Capture-Botnet-49",
        "CTU-Malware-Capture-Botnet-50",
        "CTU-Malware-Capture-Botnet-53",
    ],
    "val": [
        "CTU-Malware-Capture-Botnet-46",
        "CTU-Malware-Capture-Botnet-54",
    ],
    "test": [
        "CTU-Malware-Capture-Botnet-44",
        "CTU-Malware-Capture-Botnet-45",
        "CTU-Malware-Capture-Botnet-48",
        "CTU-Malware-Capture-Botnet-51",
        "CTU-Malware-Capture-Botnet-52",
    ],
}

# More train-heavy fixed split.
BALANCED_SPLIT = {
    "train": [
        "CTU-Malware-Capture-Botnet-42",
        "CTU-Malware-Capture-Botnet-43",
        "CTU-Malware-Capture-Botnet-44",
        "CTU-Malware-Capture-Botnet-45",
        "CTU-Malware-Capture-Botnet-46",
        "CTU-Malware-Capture-Botnet-47",
        "CTU-Malware-Capture-Botnet-50",
        "CTU-Malware-Capture-Botnet-51",
        "CTU-Malware-Capture-Botnet-52",
        "CTU-Malware-Capture-Botnet-54",
    ],
    "val": ["CTU-Malware-Capture-Botnet-48"],
    "test": ["CTU-Malware-Capture-Botnet-49", "CTU-Malware-Capture-Botnet-53"],
}

MANIFEST_FIELDS = [
    "scenario_name",
    "family",
    "graph_path",
    "graph_relpath",
    "source_file",
    "window_start",
    "window_end",
    "num_nodes",
    "num_edges",
    "num_node_features",
    "num_edge_features",
    "num_labeled_nodes",
    "num_positive_nodes",
    "num_negative_nodes",
    "num_unknown_nodes",
]


@dataclass
class GraphRecord:
    scenario_name: str
    family: str
    graph_path: str
    graph_relpath: str
    source_file: str
    window_start: str
    window_end: str
    num_nodes: int
    num_edges: int
    num_node_features: int
    num_edge_features: int
    num_labeled_nodes: int
    num_positive_nodes: int
    num_negative_nodes: int
    num_unknown_nodes: int

    def to_row(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "family": self.family,
            "graph_path": self.graph_path,
            "graph_relpath": self.graph_relpath,
            "source_file": self.source_file,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_node_features": self.num_node_features,
            "num_edge_features": self.num_edge_features,
            "num_labeled_nodes": self.num_labeled_nodes,
            "num_positive_nodes": self.num_positive_nodes,
            "num_negative_nodes": self.num_negative_nodes,
            "num_unknown_nodes": self.num_unknown_nodes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CTU-13 handoff metadata and split manifests from .pt graphs only."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing .pt graph files. Can be the ct13_pyg root or any parent folder.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where manifests, split files, and summaries will be written.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Optional graph glob(s) to scan recursively. Default: *.pt",
    )
    parser.add_argument(
        "--path-mode",
        choices=["absolute", "relative"],
        default="relative",
        help="How to store graph_path in the CSVs. graph_relpath is always written too.",
    )
    parser.add_argument(
        "--family-map",
        default=None,
        help="Optional JSON file mapping scenario_name -> family name.",
    )
    parser.add_argument(
        "--write-fixed-split",
        action="store_true",
        help="Write split_train.csv, split_val.csv, split_test.csv from a fixed scenario assignment.",
    )
    parser.add_argument(
        "--split-profile",
        choices=["strict", "balanced"],
        default="strict",
        help="Built-in fixed split profile to use when --split-json is not provided.",
    )
    parser.add_argument(
        "--split-json",
        default=None,
        help="Optional JSON file with explicit fixed split scenario assignment: {train:[...], val:[...], test:[...]}.",
    )
    parser.add_argument(
        "--write-family-pair-folds",
        action="store_true",
        help="Write grouped-CV manifests for every distinct (val_family, test_family) pair.",
    )
    parser.add_argument(
        "--folds-dirname",
        default="folds_family_pair",
        help="Subdirectory name under output-dir for family-pair fold manifests.",
    )
    parser.add_argument(
        "--allow-unknown-family",
        action="store_true",
        help="Allow scenarios not present in the family map. They will be labeled as UNKNOWN.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while scanning and writing outputs.",
    )
    return parser.parse_args()


def load_family_map(path: Optional[str]) -> Dict[str, str]:
    family_map = dict(SCENARIO_FAMILY_MAP)
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            custom = json.load(fh)
        if not isinstance(custom, dict):
            raise ValueError("family map JSON must be an object mapping scenario_name -> family")
        for key, value in custom.items():
            family_map[str(key)] = str(value)
    return family_map


def load_fixed_split(profile: str, split_json: Optional[str]) -> Dict[str, List[str]]:
    if split_json:
        with open(split_json, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        for key in ("train", "val", "test"):
            if key not in payload or not isinstance(payload[key], list):
                raise ValueError("split JSON must contain train/val/test arrays")
        return {
            "train": [str(x) for x in payload["train"]],
            "val": [str(x) for x in payload["val"]],
            "test": [str(x) for x in payload["test"]],
        }
    return STRICT_SPLIT if profile == "strict" else BALANCED_SPLIT


def ensure_dirs(output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, metadata_dir


def find_pt_files(input_root: Path, patterns: Sequence[str]) -> List[Path]:
    seen = set()
    results: List[Path] = []
    for pattern in patterns:
        for path in input_root.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                results.append(path)
    results.sort()
    return results


def infer_scenario_name(graph: Any, path: Path) -> str:
    scenario_attr = getattr(graph, "scenario_name", None)
    if isinstance(scenario_attr, str) and scenario_attr.strip():
        return scenario_attr.strip()

    for part in path.parts:
        if part.startswith("CTU-Malware-Capture-Botnet-"):
            return part

    stem = path.stem
    if stem.endswith(".pt"):
        stem = Path(stem).stem
    if "__" in stem:
        candidate = stem.split("__", 1)[0]
        if candidate.startswith("CTU-Malware-Capture-Botnet-"):
            return candidate
    raise ValueError(f"Could not infer scenario name for graph: {path}")


def shape2(value: Any) -> Tuple[int, int]:
    if value is None:
        return 0, 0
    if hasattr(value, "shape"):
        shape = list(value.shape)
        if len(shape) == 1:
            return int(shape[0]), 0
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    if isinstance(value, list):
        rows = len(value)
        cols = len(value[0]) if rows and isinstance(value[0], list) else 0
        return rows, cols
    return 0, 0


def count_labels(y: Any, labeled_mask: Any) -> Tuple[int, int, int, int]:
    y_list = [int(v) for v in y.tolist()] if hasattr(y, "tolist") else [int(v) for v in y]
    if labeled_mask is None:
        mask_list = [yy >= 0 for yy in y_list]
    else:
        mask_list = [bool(v) for v in (labeled_mask.tolist() if hasattr(labeled_mask, "tolist") else labeled_mask)]

    if len(mask_list) != len(y_list):
        raise ValueError("labeled_mask length does not match y length")

    labeled = 0
    positive = 0
    negative = 0
    unknown = 0
    for yy, mm in zip(y_list, mask_list):
        if mm and yy >= 0:
            labeled += 1
            if yy == 1:
                positive += 1
            elif yy == 0:
                negative += 1
            else:
                unknown += 1
        else:
            unknown += 1
    return labeled, positive, negative, unknown


def load_graph_record(
    pt_path: Path,
    input_root: Path,
    family_map: Dict[str, str],
    allow_unknown_family: bool,
    path_mode: str,
) -> Tuple[GraphRecord, List[str], List[str]]:
    graph = torch.load(pt_path, map_location="cpu", weights_only=False)
    scenario_name = infer_scenario_name(graph, pt_path)
    family = family_map.get(scenario_name)
    if family is None:
        if allow_unknown_family:
            family = "UNKNOWN"
        else:
            raise ValueError(
                f"Scenario '{scenario_name}' is not present in the family map. "
                f"Use --family-map or --allow-unknown-family."
            )

    num_nodes, num_node_features = shape2(getattr(graph, "x", None))
    edge_rows, edge_cols = shape2(getattr(graph, "edge_attr", None))
    if edge_rows == 0 and hasattr(graph, "edge_index") and getattr(graph.edge_index, "shape", None) is not None:
        edge_rows = int(graph.edge_index.shape[1])
    num_edges = int(getattr(graph, "num_edges_original", edge_rows))
    if num_edges == 0:
        num_edges = edge_rows

    y = getattr(graph, "y", None)
    if y is None:
        raise ValueError(f"Graph missing 'y': {pt_path}")
    labeled_mask = getattr(graph, "labeled_mask", None)
    labeled, positive, negative, unknown = count_labels(y, labeled_mask)

    graph_relpath = str(pt_path.relative_to(input_root))
    graph_path = str(pt_path.resolve()) if path_mode == "absolute" else graph_relpath
    source_file = str(getattr(graph, "source_file", ""))
    window_start = str(getattr(graph, "window_start", ""))
    window_end = str(getattr(graph, "window_end", ""))

    node_feature_names = getattr(graph, "node_feature_names", []) or []
    edge_feature_names = getattr(graph, "edge_feature_names", []) or []

    record = GraphRecord(
        scenario_name=scenario_name,
        family=family,
        graph_path=graph_path,
        graph_relpath=graph_relpath,
        source_file=source_file,
        window_start=window_start,
        window_end=window_end,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_node_features=num_node_features,
        num_edge_features=edge_cols,
        num_labeled_nodes=labeled,
        num_positive_nodes=positive,
        num_negative_nodes=negative,
        num_unknown_nodes=unknown,
    )
    return record, [str(x) for x in node_feature_names], [str(x) for x in edge_feature_names]


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_records(records: Sequence[GraphRecord]) -> Dict[str, Any]:
    by_family = Counter(r.family for r in records)
    by_scenario = Counter(r.scenario_name for r in records)
    return {
        "num_graphs": len(records),
        "num_scenarios": len(by_scenario),
        "num_families": len(by_family),
        "graphs_per_family": dict(sorted(by_family.items())),
        "graphs_per_scenario": dict(sorted(by_scenario.items())),
        "total_labeled_nodes": sum(r.num_labeled_nodes for r in records),
        "total_positive_nodes": sum(r.num_positive_nodes for r in records),
        "total_negative_nodes": sum(r.num_negative_nodes for r in records),
        "total_unknown_nodes": sum(r.num_unknown_nodes for r in records),
    }


def validate_feature_name_consistency(feature_name_lists: Sequence[List[str]], kind: str) -> List[str]:
    non_empty = [lst for lst in feature_name_lists if lst]
    if not non_empty:
        return []
    first = non_empty[0]
    for idx, lst in enumerate(non_empty[1:], start=2):
        if lst != first:
            raise ValueError(f"Inconsistent {kind} feature names across graphs (first mismatch at graph #{idx})")
    return first


def rows_for_scenarios(records: Sequence[GraphRecord], scenarios: Sequence[str], split_name: str) -> List[Dict[str, Any]]:
    wanted = set(scenarios)
    rows = []
    for record in records:
        if record.scenario_name in wanted:
            row = record.to_row()
            row["split"] = split_name
            rows.append(row)
    return rows


def write_fixed_split(
    records: Sequence[GraphRecord],
    fixed_split: Dict[str, List[str]],
    metadata_dir: Path,
) -> Dict[str, Any]:
    rows_train = rows_for_scenarios(records, fixed_split["train"], "train")
    rows_val = rows_for_scenarios(records, fixed_split["val"], "val")
    rows_test = rows_for_scenarios(records, fixed_split["test"], "test")

    split_fieldnames = list(MANIFEST_FIELDS) + ["split"]
    write_csv(metadata_dir / "split_train.csv", rows_train, split_fieldnames)
    write_csv(metadata_dir / "split_val.csv", rows_val, split_fieldnames)
    write_csv(metadata_dir / "split_test.csv", rows_test, split_fieldnames)

    split_json = {
        "train": fixed_split["train"],
        "val": fixed_split["val"],
        "test": fixed_split["test"],
    }
    with (metadata_dir / "splits.json").open("w", encoding="utf-8") as fh:
        json.dump(split_json, fh, indent=2)

    return {
        "train_graphs": len(rows_train),
        "val_graphs": len(rows_val),
        "test_graphs": len(rows_test),
        "train_scenarios": fixed_split["train"],
        "val_scenarios": fixed_split["val"],
        "test_scenarios": fixed_split["test"],
    }


def write_family_pair_folds(records: Sequence[GraphRecord], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    family_to_records: Dict[str, List[GraphRecord]] = defaultdict(list)
    for record in records:
        family_to_records[record.family].append(record)

    families = sorted(family_to_records.keys())
    fold_summaries: List[Dict[str, Any]] = []
    split_fieldnames = list(MANIFEST_FIELDS) + ["split"]
    fold_id = 0

    for test_family in families:
        for val_family in families:
            if val_family == test_family:
                continue
            train_families = [fam for fam in families if fam not in {val_family, test_family}]
            if not train_families:
                continue

            fold_id += 1
            fold_name = f"fold_{fold_id:03d}__val-{val_family}__test-{test_family}"
            fold_dir = out_dir / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_rows: List[Dict[str, Any]] = []
            val_rows: List[Dict[str, Any]] = []
            test_rows: List[Dict[str, Any]] = []

            for fam in train_families:
                for record in family_to_records[fam]:
                    row = record.to_row()
                    row["split"] = "train"
                    train_rows.append(row)
            for record in family_to_records[val_family]:
                row = record.to_row()
                row["split"] = "val"
                val_rows.append(row)
            for record in family_to_records[test_family]:
                row = record.to_row()
                row["split"] = "test"
                test_rows.append(row)

            write_csv(fold_dir / "train.csv", train_rows, split_fieldnames)
            write_csv(fold_dir / "val.csv", val_rows, split_fieldnames)
            write_csv(fold_dir / "test.csv", test_rows, split_fieldnames)
            with (fold_dir / "fold.json").open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "train_families": train_families,
                        "val_family": val_family,
                        "test_family": test_family,
                        "train_graphs": len(train_rows),
                        "val_graphs": len(val_rows),
                        "test_graphs": len(test_rows),
                    },
                    fh,
                    indent=2,
                )

            fold_summaries.append(
                {
                    "fold_name": fold_name,
                    "val_family": val_family,
                    "test_family": test_family,
                    "train_families": train_families,
                    "train_graphs": len(train_rows),
                    "val_graphs": len(val_rows),
                    "test_graphs": len(test_rows),
                }
            )

    with (out_dir / "folds_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(fold_summaries, fh, indent=2)

    return {
        "num_folds": len(fold_summaries),
        "families": families,
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    _, metadata_dir = ensure_dirs(output_dir)

    family_map = load_family_map(args.family_map)
    patterns = args.glob if args.glob else ["*.pt"]
    pt_files = find_pt_files(input_root, patterns)
    if not pt_files:
        raise FileNotFoundError(f"No .pt graphs found under {input_root}")

    records: List[GraphRecord] = []
    node_feature_name_lists: List[List[str]] = []
    edge_feature_name_lists: List[List[str]] = []

    for idx, pt_path in enumerate(pt_files, start=1):
        record, node_feature_names, edge_feature_names = load_graph_record(
            pt_path=pt_path,
            input_root=input_root,
            family_map=family_map,
            allow_unknown_family=args.allow_unknown_family,
            path_mode=args.path_mode,
        )
        records.append(record)
        node_feature_name_lists.append(node_feature_names)
        edge_feature_name_lists.append(edge_feature_names)
        if args.verbose and (idx == 1 or idx % 100 == 0 or idx == len(pt_files)):
            print(f"Scanned {idx}/{len(pt_files)} graphs")

    records.sort(key=lambda r: (r.scenario_name, r.window_start, r.graph_relpath))
    write_csv(metadata_dir / "graphs_index_pyg.csv", (r.to_row() for r in records), MANIFEST_FIELDS)

    node_feature_names = validate_feature_name_consistency(node_feature_name_lists, "node")
    edge_feature_names = validate_feature_name_consistency(edge_feature_name_lists, "edge")
    with (metadata_dir / "feature_names.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "node_feature_names": node_feature_names,
                "edge_feature_names": edge_feature_names,
            },
            fh,
            indent=2,
        )

    summary: Dict[str, Any] = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "path_mode": args.path_mode,
        "master_manifest": str(metadata_dir / "graphs_index_pyg.csv"),
        "dataset_summary": summarize_records(records),
    }

    if args.write_fixed_split:
        fixed_split = load_fixed_split(args.split_profile, args.split_json)
        summary["fixed_split"] = write_fixed_split(records, fixed_split, metadata_dir)
        summary["fixed_split"]["profile"] = args.split_profile if not args.split_json else "custom"

    if args.write_family_pair_folds:
        folds_dir = output_dir / args.folds_dirname
        summary["family_pair_folds"] = write_family_pair_folds(records, folds_dir)
        summary["family_pair_folds"]["folds_dir"] = str(folds_dir)

    with (metadata_dir / "handoff_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if args.verbose:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Wrote master manifest: {metadata_dir / 'graphs_index_pyg.csv'}")
        if args.write_fixed_split:
            print(f"Wrote fixed split manifests under: {metadata_dir}")
        if args.write_family_pair_folds:
            print(f"Wrote family-pair CV folds under: {output_dir / args.folds_dirname}")


if __name__ == "__main__":
    main()
