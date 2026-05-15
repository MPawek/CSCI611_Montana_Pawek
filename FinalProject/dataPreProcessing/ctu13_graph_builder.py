#!/usr/bin/env python3
"""
Build 5-minute communication graphs from the CTU-13 dataset for GNN training.

What this script does:
- Recursively finds CTU-13 .binetflow files under an input directory.
- Loads each bidirectional flow file.
- Splits traffic into fixed time windows (default: 5 minutes).
- Builds one directed communication graph per window.
- Aggregates flow statistics into edge features and host statistics into node features.
- Assigns node labels for botnet detection using scenario README host labels when
  available, with a safe fallback to flow-level labels.
- Saves each graph as a torch_geometric.data.Data object (.pt) plus an index CSV.

Recommended CTU-13 input:
    .../<scenario>/detailed-bidirectional-flow-labels/*.binetflow

Example:
    python ctu13_graph_builder.py \
        --input-root /data/CTU-13-Dataset \
        --output-dir /data/ctu13_graphs \
        --window-minutes 5 \
        --label-strategy hybrid

Output structure:
    output-dir/
      graphs/<scenario>/<scenario>__YYYYmmdd_HHMMSS.pt
      metadata/graphs_index.csv
      metadata/feature_names.json

Each saved PyG Data object contains:
    x             [num_nodes, num_node_features]    float32
    edge_index    [2, num_edges]                    int64
    edge_attr     [num_edges, num_edge_features]    float32
    y             [num_nodes]                       int64  (1=botnet, 0=normal, -1=ignore)
    labeled_mask  [num_nodes]                       bool
    node_id       [num_nodes]                       int64  (0..num_nodes-1)
    scenario_name str
    source_file   str
    window_start  int (unix seconds)
    window_end    int (unix seconds)

Notes:
- Features are log1p-transformed inside the script to tame heavy-tailed counts.
- Do standardization/z-scoring later, after you split by scenario, using train only.
- The script does not create train/val/test splits; CTU-13 should be split by scenario.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


REQUIRED_COLUMN_GROUPS = {
    "time": ["StartTime", "starttime", "start_time", "stime", "time"],
    "src": ["SrcAddr", "srcaddr", "src_ip", "source", "src"],
    "dst": ["DstAddr", "dstaddr", "dst_ip", "destination", "dst"],
    "proto": ["Proto", "proto", "protocol"],
    "sport": ["Sport", "sport", "src_port", "source_port"],
    "dport": ["Dport", "dport", "dst_port", "destination_port"],
    "dur": ["Dur", "dur", "duration"],
    "totpkts": ["TotPkts", "totpkts", "packets", "pkts", "total_packets"],
    "totbytes": ["TotBytes", "totbytes", "bytes", "total_bytes"],
    "srcbytes": ["SrcBytes", "srcbytes", "src_bytes", "source_bytes"],
    "label": ["Label", "label", "class"],
    "dir": ["Dir", "dir", "direction"],
    "state": ["State", "state"],
}

NODE_FEATURE_NAMES = [
    "out_degree",
    "in_degree",
    "out_flows",
    "in_flows",
    "out_bytes",
    "in_bytes",
    "out_pkts",
    "in_pkts",
    "out_srcbytes",
    "in_srcbytes",
    "out_unique_peers",
    "in_unique_peers",
    "out_unique_dports",
    "in_unique_sports",
    "out_mean_dur",
    "in_mean_dur",
    "out_tcp",
    "out_udp",
    "out_icmp",
    "out_other_proto",
    "in_tcp",
    "in_udp",
    "in_icmp",
    "in_other_proto",
]

EDGE_FEATURE_NAMES = [
    "n_flows",
    "sum_pkts",
    "sum_bytes",
    "sum_srcbytes",
    "mean_dur",
    "uniq_sports",
    "uniq_dports",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "proto_other",
]


@dataclass
class ScenarioLabels:
    infected_ips: Set[str]
    normal_ips: Set[str]


@dataclass
class BuildStats:
    scenarios_seen: int = 0
    windows_seen: int = 0
    graphs_saved: int = 0
    graphs_skipped_no_edges: int = 0
    graphs_skipped_min_nodes: int = 0
    graphs_skipped_min_labeled: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CTU-13 time-window communication graphs for GNN training.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing CTU-13 scenario folders or .binetflow files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where graphs and metadata will be written.")
    parser.add_argument("--window-minutes", type=int, default=5, help="Window size in minutes. Default: 5")
    parser.add_argument("--label-strategy", choices=["readme", "flow", "hybrid"], default="hybrid",
                        help="Node labeling strategy. 'hybrid' uses scenario README IP labels when present, then falls back to flow labels.")
    parser.add_argument("--min-nodes", type=int, default=2, help="Skip graphs with fewer than this many nodes. Default: 2")
    parser.add_argument("--min-labeled-nodes", type=int, default=1, help="Skip graphs with fewer than this many labeled nodes. Default: 1")
    parser.add_argument("--file-pattern", type=str, default="*.binetflow", help="Glob pattern for CTU-13 flow files. Default: *.binetflow")
    parser.add_argument("--verbose", action="store_true", help="Print progress per scenario/file.")
    return parser.parse_args()


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def ensure_dirs(output_dir: Path) -> Tuple[Path, Path]:
    graphs_dir = output_dir / "graphs"
    meta_dir = output_dir / "metadata"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir, meta_dir


def discover_binetflows(input_root: Path, pattern: str) -> List[Path]:
    if input_root.is_file() and input_root.suffix.lower() == ".binetflow":
        return [input_root]
    return sorted([p for p in input_root.rglob(pattern) if p.is_file()])


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    existing = {c.lower(): c for c in df.columns}
    for canonical, aliases in REQUIRED_COLUMN_GROUPS.items():
        for alias in aliases:
            found = existing.get(alias.lower())
            if found is not None:
                rename_map[found] = canonical
                break
    df = df.rename(columns=rename_map)
    required_now = ["time", "src", "dst", "proto", "sport", "dport", "dur", "totpkts", "totbytes", "srcbytes", "label"]
    missing = [c for c in required_now if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}. Columns seen: {list(df.columns)}")
    # Optional columns
    for optional in ["dir", "state"]:
        if optional not in df.columns:
            df[optional] = ""
    return df


def read_flow_file(path: Path) -> pd.DataFrame:
    # sep=None with the Python engine lets pandas sniff common delimiters.
    df = pd.read_csv(path, sep=None, engine="python")
    df = canonicalize_columns(df)

    # Timestamp parsing: first plain parse, then a dayfirst fallback for stubborn files.
    ts = pd.to_datetime(df["time"], errors="coerce")
    if ts.isna().mean() > 0.25:
        ts = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)
    if ts.isna().all():
        raise ValueError(f"Could not parse any timestamps from {path}")
    df["time"] = ts
    df = df.loc[~df["time"].isna()].copy()

    # Standardize text columns.
    for col in ["src", "dst", "proto", "sport", "dport", "label", "dir", "state"]:
        df[col] = df[col].astype(str).str.strip()

    # Numeric cleanup.
    for col in ["dur", "totpkts", "totbytes", "srcbytes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[col] = df[col].clip(lower=0.0)

    # Lower-cased protocol for feature extraction.
    df["proto_norm"] = df["proto"].str.lower()
    return df


def scenario_name_for_file(path: Path) -> str:
    # Expected CTU-13 layout: <scenario>/detailed-bidirectional-flow-labels/<file>.binetflow
    parents = list(path.parents)
    for parent in parents:
        if parent.name.startswith("CTU-Malware-Capture-Botnet-"):
            return parent.name
    # Fallback: use file stem when directory naming is different.
    return path.stem


def scenario_dir_for_file(path: Path) -> Path:
    for parent in path.parents:
        if parent.name.startswith("CTU-Malware-Capture-Botnet-"):
            return parent
    return path.parent


_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def parse_scenario_readme(scenario_dir: Path) -> ScenarioLabels:
    infected: Set[str] = set()
    normal: Set[str] = set()

    readme_candidates = []
    for name in ["README", "README.txt", "README.md", "README.html"]:
        p = scenario_dir / name
        if p.exists():
            readme_candidates.append(p)
    # Also look one level deeper if the bundle layout is unusual.
    readme_candidates.extend([p for p in scenario_dir.glob("**/README*") if p.is_file()])

    if not readme_candidates:
        return ScenarioLabels(infected_ips=infected, normal_ips=normal)

    readme_path = sorted(set(readme_candidates), key=lambda p: len(str(p)))[0]
    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    section: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        low = line.lower()
        if "infected hosts" in low:
            section = "infected"
            continue
        if "normal hosts" in low:
            section = "normal"
            continue
        if line.startswith("##") or line.startswith("# "):
            section = None
            continue

        ips = _IP_RE.findall(line)
        if not ips:
            continue

        if section == "infected":
            infected.update(ips)
        elif section == "normal":
            normal.update(ips)
        else:
            # Weak fallback when section parsing is not enough.
            if "label: botnet" in low:
                infected.update(ips)
            elif "label:" in low and ("normal" in low or "webserver" in low or "dns-server" in low or "matlab-server" in low):
                normal.update(ips)

    return ScenarioLabels(infected_ips=infected, normal_ips=normal)


def flow_label_to_class(label: str) -> int:
    label = str(label).strip().lower()
    if "from-botnet" in label:
        return 1
    if "from-normal" in label:
        return 0
    return -1


def protocol_buckets(proto_series: pd.Series) -> pd.DataFrame:
    p = proto_series.astype(str).str.lower()
    return pd.DataFrame({
        "tcp": (p == "tcp").astype(np.int64),
        "udp": (p == "udp").astype(np.int64),
        "icmp": (p == "icmp").astype(np.int64),
        "other": (~p.isin(["tcp", "udp", "icmp"])).astype(np.int64),
    }, index=proto_series.index)


def safe_nunique(series: pd.Series) -> int:
    s = series.astype(str)
    s = s[~s.isin(["", "nan", "None"])]
    return int(s.nunique())


def build_edge_table(dfw: pd.DataFrame) -> pd.DataFrame:
    proto_df = protocol_buckets(dfw["proto_norm"])
    tmp = dfw[["src", "dst", "sport", "dport", "dur", "totpkts", "totbytes", "srcbytes"]].copy()
    tmp = pd.concat([tmp, proto_df], axis=1)

    edge_df = tmp.groupby(["src", "dst"], as_index=False).agg(
        n_flows=("src", "size"),
        sum_pkts=("totpkts", "sum"),
        sum_bytes=("totbytes", "sum"),
        sum_srcbytes=("srcbytes", "sum"),
        mean_dur=("dur", "mean"),
        uniq_sports=("sport", safe_nunique),
        uniq_dports=("dport", safe_nunique),
        proto_tcp=("tcp", "sum"),
        proto_udp=("udp", "sum"),
        proto_icmp=("icmp", "sum"),
        proto_other=("other", "sum"),
    )
    edge_df[EDGE_FEATURE_NAMES] = edge_df[EDGE_FEATURE_NAMES].astype(np.float32)
    return edge_df


def build_node_table(dfw: pd.DataFrame, edge_df: pd.DataFrame, node_ips: Sequence[str]) -> pd.DataFrame:
    proto_df = protocol_buckets(dfw["proto_norm"])
    flow_df = pd.concat([dfw[["src", "dst", "sport", "dport", "dur", "totpkts", "totbytes", "srcbytes"]].copy(), proto_df], axis=1)

    out_agg = flow_df.groupby("src").agg(
        out_flows=("src", "size"),
        out_bytes=("totbytes", "sum"),
        out_pkts=("totpkts", "sum"),
        out_srcbytes=("srcbytes", "sum"),
        out_unique_peers=("dst", "nunique"),
        out_unique_dports=("dport", safe_nunique),
        out_mean_dur=("dur", "mean"),
        out_tcp=("tcp", "sum"),
        out_udp=("udp", "sum"),
        out_icmp=("icmp", "sum"),
        out_other_proto=("other", "sum"),
    )

    in_agg = flow_df.groupby("dst").agg(
        in_flows=("dst", "size"),
        in_bytes=("totbytes", "sum"),
        in_pkts=("totpkts", "sum"),
        in_srcbytes=("srcbytes", "sum"),
        in_unique_peers=("src", "nunique"),
        in_unique_sports=("sport", safe_nunique),
        in_mean_dur=("dur", "mean"),
        in_tcp=("tcp", "sum"),
        in_udp=("udp", "sum"),
        in_icmp=("icmp", "sum"),
        in_other_proto=("other", "sum"),
    )

    out_deg = edge_df.groupby("src").agg(out_degree=("dst", "nunique"))
    in_deg = edge_df.groupby("dst").agg(in_degree=("src", "nunique"))

    node_df = pd.DataFrame(index=pd.Index(node_ips, name="ip"))
    node_df = node_df.join(out_deg, how="left")
    node_df = node_df.join(in_deg, how="left")
    node_df = node_df.join(out_agg, how="left")
    node_df = node_df.join(in_agg, how="left")
    node_df = node_df.fillna(0.0)

    # Guarantee column order.
    node_df = node_df.reindex(columns=NODE_FEATURE_NAMES).fillna(0.0)
    return node_df


def label_nodes(
    dfw: pd.DataFrame,
    node_ips: Sequence[str],
    scenario_labels: ScenarioLabels,
    label_strategy: str,
) -> np.ndarray:
    flow_y = dfw["label"].map(flow_label_to_class)
    pos_sources = set(dfw.loc[flow_y == 1, "src"].astype(str))
    neg_sources = set(dfw.loc[flow_y == 0, "src"].astype(str))

    y: List[int] = []
    for ip in node_ips:
        value = -1
        if label_strategy in {"readme", "hybrid"}:
            if ip in scenario_labels.infected_ips:
                value = 1
            elif ip in scenario_labels.normal_ips:
                value = 0

        if value == -1 and label_strategy in {"flow", "hybrid"}:
            if ip in pos_sources:
                value = 1
            elif ip in neg_sources:
                value = 0

        y.append(value)
    return np.asarray(y, dtype=np.int64)


def make_pyg_data(
    dfw: pd.DataFrame,
    scenario_name: str,
    source_file: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    scenario_labels: ScenarioLabels,
    label_strategy: str,
) -> Tuple[Data, List[str]]:
    edge_df = build_edge_table(dfw)
    node_ips = sorted(set(dfw["src"].astype(str)).union(set(dfw["dst"].astype(str))))
    ip_to_idx = {ip: idx for idx, ip in enumerate(node_ips)}

    node_df = build_node_table(dfw, edge_df, node_ips)
    x = np.log1p(node_df.to_numpy(dtype=np.float32))

    edge_index = np.vstack([
        edge_df["src"].map(ip_to_idx).to_numpy(dtype=np.int64),
        edge_df["dst"].map(ip_to_idx).to_numpy(dtype=np.int64),
    ])
    edge_attr = np.log1p(edge_df[EDGE_FEATURE_NAMES].to_numpy(dtype=np.float32))

    y = label_nodes(dfw, node_ips, scenario_labels, label_strategy)
    labeled_mask = y >= 0

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(y),
        labeled_mask=torch.from_numpy(labeled_mask),
        node_id=torch.arange(len(node_ips), dtype=torch.long),
    )
    data.scenario_name = scenario_name
    data.source_file = source_file
    data.window_start = int(window_start.timestamp())
    data.window_end = int(window_end.timestamp())
    return data, node_ips


def write_feature_metadata(meta_dir: Path) -> None:
    payload = {
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "label_meaning": {"1": "botnet/infected", "0": "normal", "-1": "ignore/unlabeled"},
    }
    (meta_dir / "feature_names.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_file(
    binetflow_path: Path,
    graphs_dir: Path,
    meta_rows: List[Dict[str, object]],
    stats: BuildStats,
    window_minutes: int,
    label_strategy: str,
    min_nodes: int,
    min_labeled_nodes: int,
    verbose: bool,
) -> None:
    scenario_name = scenario_name_for_file(binetflow_path)
    scenario_dir = scenario_dir_for_file(binetflow_path)
    scenario_labels = parse_scenario_readme(scenario_dir)
    stats.scenarios_seen += 1

    df = read_flow_file(binetflow_path)
    df["window_start"] = df["time"].dt.floor(f"{window_minutes}min")

    scenario_graph_dir = graphs_dir / scenario_name
    scenario_graph_dir.mkdir(parents=True, exist_ok=True)

    for window_start, dfw in df.groupby("window_start", sort=True):
        stats.windows_seen += 1
        if dfw.empty:
            continue

        window_end = window_start + pd.Timedelta(minutes=window_minutes)
        data, node_ips = make_pyg_data(
            dfw=dfw,
            scenario_name=scenario_name,
            source_file=str(binetflow_path),
            window_start=window_start,
            window_end=window_end,
            scenario_labels=scenario_labels,
            label_strategy=label_strategy,
        )

        num_nodes = int(data.x.size(0))
        num_edges = int(data.edge_index.size(1))
        num_labeled = int(data.labeled_mask.sum().item())
        num_positive = int(((data.y == 1) & data.labeled_mask).sum().item())

        if num_edges == 0:
            stats.graphs_skipped_no_edges += 1
            continue
        if num_nodes < min_nodes:
            stats.graphs_skipped_min_nodes += 1
            continue
        if num_labeled < min_labeled_nodes:
            stats.graphs_skipped_min_labeled += 1
            continue

        stamp = pd.Timestamp(window_start).strftime("%Y%m%d_%H%M%S")
        graph_path = scenario_graph_dir / f"{scenario_name}__{stamp}.pt"
        torch.save(data, graph_path)

        # Save node mapping as sidecar JSON so you can map predictions back to IPs later.
        node_map_path = graph_path.with_suffix(".nodes.json")
        node_map_path.write_text(json.dumps({"node_ips": node_ips}, indent=2), encoding="utf-8")

        meta_rows.append({
            "scenario_name": scenario_name,
            "source_file": str(binetflow_path),
            "graph_path": str(graph_path),
            "node_map_path": str(node_map_path),
            "window_start": pd.Timestamp(window_start).isoformat(),
            "window_end": pd.Timestamp(window_end).isoformat(),
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_labeled_nodes": num_labeled,
            "num_positive_nodes": num_positive,
            "readme_infected_ips": len(scenario_labels.infected_ips),
            "readme_normal_ips": len(scenario_labels.normal_ips),
            "label_strategy": label_strategy,
        })
        stats.graphs_saved += 1

    log(f"Processed {scenario_name}: {binetflow_path.name}", verbose)


def main() -> None:
    args = parse_args()
    graphs_dir, meta_dir = ensure_dirs(args.output_dir)
    write_feature_metadata(meta_dir)

    flow_files = discover_binetflows(args.input_root, args.file_pattern)
    if not flow_files:
        raise SystemExit(f"No flow files matching {args.file_pattern!r} found under {args.input_root}")

    stats = BuildStats()
    meta_rows: List[Dict[str, object]] = []

    for flow_file in flow_files:
        try:
            process_file(
                binetflow_path=flow_file,
                graphs_dir=graphs_dir,
                meta_rows=meta_rows,
                stats=stats,
                window_minutes=args.window_minutes,
                label_strategy=args.label_strategy,
                min_nodes=args.min_nodes,
                min_labeled_nodes=args.min_labeled_nodes,
                verbose=args.verbose,
            )
        except Exception as exc:
            print(f"[WARN] Failed to process {flow_file}: {exc}", flush=True)

    index_df = pd.DataFrame(meta_rows)
    index_path = meta_dir / "graphs_index.csv"
    index_df.to_csv(index_path, index=False)

    summary = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "window_minutes": args.window_minutes,
        "label_strategy": args.label_strategy,
        "num_flow_files_seen": len(flow_files),
        "num_graphs_saved": stats.graphs_saved,
        "num_windows_seen": stats.windows_seen,
        "graphs_skipped_no_edges": stats.graphs_skipped_no_edges,
        "graphs_skipped_min_nodes": stats.graphs_skipped_min_nodes,
        "graphs_skipped_min_labeled": stats.graphs_skipped_min_labeled,
        "node_feature_dim": len(NODE_FEATURE_NAMES),
        "edge_feature_dim": len(EDGE_FEATURE_NAMES),
    }
    (meta_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote graph index to: {index_path}", flush=True)


if __name__ == "__main__":
    main()
