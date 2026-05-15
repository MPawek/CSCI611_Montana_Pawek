from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


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

_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
TIMESTAMP_FORMATS = [
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
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
    rows_seen: int = 0
    rows_kept: int = 0
    parse_errors: int = 0
    out_of_order_rows: int = 0


@dataclass
class GraphAccumulator:
    edge_stats: Dict[Tuple[str, str], dict]
    node_stats: Dict[str, dict]
    pos_sources: Set[str]
    neg_sources: Set[str]


class MetadataWriter:
    def __init__(self, meta_dir: Path) -> None:
        self.index_path = meta_dir / "graphs_index.csv"
        self._fh = self.index_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=[
                "scenario_name",
                "source_file",
                "graph_path",
                "window_start",
                "window_end",
                "num_nodes",
                "num_edges",
                "num_labeled_nodes",
                "num_positive_nodes",
                "readme_infected_ips",
                "readme_normal_ips",
                "label_strategy",
                "output_format",
            ],
        )
        self._writer.writeheader()

    def write_row(self, row: Dict[str, object]) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CTU-13 time-window communication graphs using only the Python standard library.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory containing CTU-13 scenario folders or .binetflow files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where graphs and metadata will be written.")
    parser.add_argument("--window-minutes", type=int, default=5, help="Window size in minutes. Default: 5")
    parser.add_argument("--label-strategy", choices=["readme", "flow", "hybrid"], default="hybrid",
                        help="Node labeling strategy. 'hybrid' uses scenario README IP labels when present, then falls back to flow labels.")
    parser.add_argument("--min-nodes", type=int, default=2, help="Skip graphs with fewer than this many nodes. Default: 2")
    parser.add_argument("--min-labeled-nodes", type=int, default=1, help="Skip graphs with fewer than this many labeled nodes. Default: 1")
    parser.add_argument("--file-pattern", type=str, default="*.binetflow", help="Glob pattern for CTU-13 flow files. Default: *.binetflow")
    parser.add_argument("--output-format", choices=["json", "json.gz"], default="json.gz",
                        help="Per-window graph serialization format. Default: json.gz")
    parser.add_argument("--verbose", action="store_true", help="Print progress per scenario/file.")
    return parser.parse_args()


def log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


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


def scenario_name_for_file(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("CTU-Malware-Capture-Botnet-"):
            return parent.name
    return path.stem


def scenario_dir_for_file(path: Path) -> Path:
    for parent in path.parents:
        if parent.name.startswith("CTU-Malware-Capture-Botnet-"):
            return parent
    return path.parent


def parse_scenario_readme(scenario_dir: Path) -> ScenarioLabels:
    infected: Set[str] = set()
    normal: Set[str] = set()

    readme_candidates: List[Path] = []
    for name in ["README", "README.txt", "README.md", "README.html"]:
        p = scenario_dir / name
        if p.exists():
            readme_candidates.append(p)
    readme_candidates.extend([p for p in scenario_dir.glob("**/README*") if p.is_file()])

    if not readme_candidates:
        return ScenarioLabels(infected_ips=infected, normal_ips=normal)

    chosen = sorted(set(readme_candidates), key=lambda p: len(str(p)))[0]
    text = chosen.read_text(encoding="utf-8", errors="ignore")

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
        if line.startswith("##") or line.startswith("# ") or line.startswith("<h"):
            section = None

        ips = _IP_RE.findall(line)
        if not ips:
            continue

        if section == "infected":
            infected.update(ips)
        elif section == "normal":
            normal.update(ips)
        else:
            if "label: botnet" in low:
                infected.update(ips)
            elif "label:" in low and ("normal" in low or "webserver" in low or "dns-server" in low or "matlab-server" in low):
                normal.update(ips)

    return ScenarioLabels(infected_ips=infected, normal_ips=normal)


def flow_label_to_class(label: str) -> int:
    value = str(label).strip().lower()
    if "from-botnet" in value:
        return 1
    if "from-normal" in value:
        return 0
    return -1


def canonicalize_header(header: Sequence[str]) -> Dict[str, int]:
    lower_to_index = {str(col).strip().lower(): idx for idx, col in enumerate(header)}
    mapping: Dict[str, int] = {}

    for canonical, aliases in REQUIRED_COLUMN_GROUPS.items():
        for alias in aliases:
            idx = lower_to_index.get(alias.lower())
            if idx is not None:
                mapping[canonical] = idx
                break

    required = ["time", "src", "dst", "proto", "sport", "dport", "dur", "totpkts", "totbytes", "srcbytes", "label"]
    missing = [name for name in required if name not in mapping]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}. Header seen: {list(header)}")

    return mapping


def sniff_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        sample = fh.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def parse_float(value: str) -> float:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return 0.0
    try:
        num = float(text)
    except ValueError:
        return 0.0
    if num < 0.0:
        return 0.0
    return num


def parse_timestamp(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    # Try the built-in ISO parser first.
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        pass

    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    raise ValueError(f"unrecognized timestamp format: {value!r}")


def floor_window_start(dt: datetime, window_seconds: int) -> datetime:
    epoch = int(dt.timestamp())
    floored = (epoch // window_seconds) * window_seconds
    return datetime.fromtimestamp(floored)


def new_accumulator() -> GraphAccumulator:
    return GraphAccumulator(
        edge_stats={},
        node_stats={},
        pos_sources=set(),
        neg_sources=set(),
    )


def get_node_stat(node_stats: Dict[str, dict], ip: str) -> dict:
    node = node_stats.get(ip)
    if node is None:
        node = {
            "out_flows": 0.0,
            "in_flows": 0.0,
            "out_bytes": 0.0,
            "in_bytes": 0.0,
            "out_pkts": 0.0,
            "in_pkts": 0.0,
            "out_srcbytes": 0.0,
            "in_srcbytes": 0.0,
            "out_dur_sum": 0.0,
            "in_dur_sum": 0.0,
            "out_tcp": 0.0,
            "out_udp": 0.0,
            "out_icmp": 0.0,
            "out_other_proto": 0.0,
            "in_tcp": 0.0,
            "in_udp": 0.0,
            "in_icmp": 0.0,
            "in_other_proto": 0.0,
            "out_peers": set(),
            "in_peers": set(),
            "out_dports": set(),
            "in_sports": set(),
            "out_neighbors": set(),
            "in_neighbors": set(),
        }
        node_stats[ip] = node
    return node


def proto_bucket(proto: str) -> str:
    p = str(proto).strip().lower()
    if p == "tcp":
        return "tcp"
    if p == "udp":
        return "udp"
    if p == "icmp":
        return "icmp"
    return "other"


def safe_add_token(values: Set[str], token: str) -> None:
    t = str(token).strip()
    if t and t.lower() not in {"nan", "none"}:
        values.add(t)


def add_row_to_accumulator(acc: GraphAccumulator, row: Dict[str, object]) -> None:
    src = str(row["src"])
    dst = str(row["dst"])
    sport = str(row["sport"])
    dport = str(row["dport"])
    dur = float(row["dur"])
    totpkts = float(row["totpkts"])
    totbytes = float(row["totbytes"])
    srcbytes = float(row["srcbytes"])
    label = str(row["label"])
    proto_key = proto_bucket(str(row["proto"]))

    edge_key = (src, dst)
    edge = acc.edge_stats.get(edge_key)
    if edge is None:
        edge = {
            "n_flows": 0.0,
            "sum_pkts": 0.0,
            "sum_bytes": 0.0,
            "sum_srcbytes": 0.0,
            "dur_sum": 0.0,
            "uniq_sports": set(),
            "uniq_dports": set(),
            "proto_tcp": 0.0,
            "proto_udp": 0.0,
            "proto_icmp": 0.0,
            "proto_other": 0.0,
        }
        acc.edge_stats[edge_key] = edge

    edge["n_flows"] += 1.0
    edge["sum_pkts"] += totpkts
    edge["sum_bytes"] += totbytes
    edge["sum_srcbytes"] += srcbytes
    edge["dur_sum"] += dur
    safe_add_token(edge["uniq_sports"], sport)
    safe_add_token(edge["uniq_dports"], dport)
    edge[f"proto_{proto_key}"] += 1.0

    src_node = get_node_stat(acc.node_stats, src)
    src_node["out_flows"] += 1.0
    src_node["out_bytes"] += totbytes
    src_node["out_pkts"] += totpkts
    src_node["out_srcbytes"] += srcbytes
    src_node["out_dur_sum"] += dur
    src_node[f"out_{proto_key}" if proto_key != "other" else "out_other_proto"] += 1.0
    src_node["out_peers"].add(dst)
    src_node["out_neighbors"].add(dst)
    safe_add_token(src_node["out_dports"], dport)

    dst_node = get_node_stat(acc.node_stats, dst)
    dst_node["in_flows"] += 1.0
    dst_node["in_bytes"] += totbytes
    dst_node["in_pkts"] += totpkts
    dst_node["in_srcbytes"] += srcbytes
    dst_node["in_dur_sum"] += dur
    dst_node[f"in_{proto_key}" if proto_key != "other" else "in_other_proto"] += 1.0
    dst_node["in_peers"].add(src)
    dst_node["in_neighbors"].add(src)
    safe_add_token(dst_node["in_sports"], sport)

    flow_class = flow_label_to_class(label)
    if flow_class == 1:
        acc.pos_sources.add(src)
    elif flow_class == 0:
        acc.neg_sources.add(src)


def label_nodes(node_ips: Sequence[str], acc: GraphAccumulator, scenario_labels: ScenarioLabels, label_strategy: str) -> List[int]:
    labels: List[int] = []
    for ip in node_ips:
        value = -1

        if label_strategy in {"readme", "hybrid"}:
            if ip in scenario_labels.infected_ips:
                value = 1
            elif ip in scenario_labels.normal_ips:
                value = 0

        if value == -1 and label_strategy in {"flow", "hybrid"}:
            if ip in acc.pos_sources:
                value = 1
            elif ip in acc.neg_sources:
                value = 0

        labels.append(value)
    return labels


def log1p_list(values: Iterable[float]) -> List[float]:
    return [math.log1p(max(0.0, float(v))) for v in values]


def write_json_payload(path: Path, payload: dict, output_format: str) -> None:
    if output_format == "json.gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
    else:
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def finalize_window(
    acc: GraphAccumulator,
    scenario_name: str,
    source_file: Path,
    window_start: datetime,
    window_seconds: int,
    graphs_dir: Path,
    metadata_writer: MetadataWriter,
    scenario_labels: ScenarioLabels,
    label_strategy: str,
    min_nodes: int,
    min_labeled_nodes: int,
    output_format: str,
    stats: BuildStats,
) -> None:
    stats.windows_seen += 1

    if not acc.edge_stats:
        stats.graphs_skipped_no_edges += 1
        return

    node_ips = sorted(acc.node_stats.keys())
    num_nodes = len(node_ips)
    if num_nodes < min_nodes:
        stats.graphs_skipped_min_nodes += 1
        return

    ip_to_idx = {ip: idx for idx, ip in enumerate(node_ips)}
    labels = label_nodes(node_ips, acc, scenario_labels, label_strategy)
    labeled_mask = [y >= 0 for y in labels]
    num_labeled = sum(1 for value in labeled_mask if value)
    if num_labeled < min_labeled_nodes:
        stats.graphs_skipped_min_labeled += 1
        return

    x: List[List[float]] = []
    for ip in node_ips:
        node = acc.node_stats[ip]
        out_flows = node["out_flows"]
        in_flows = node["in_flows"]
        out_mean_dur = node["out_dur_sum"] / out_flows if out_flows else 0.0
        in_mean_dur = node["in_dur_sum"] / in_flows if in_flows else 0.0
        features = [
            float(len(node["out_neighbors"])),
            float(len(node["in_neighbors"])),
            node["out_flows"],
            node["in_flows"],
            node["out_bytes"],
            node["in_bytes"],
            node["out_pkts"],
            node["in_pkts"],
            node["out_srcbytes"],
            node["in_srcbytes"],
            float(len(node["out_peers"])),
            float(len(node["in_peers"])),
            float(len(node["out_dports"])),
            float(len(node["in_sports"])),
            out_mean_dur,
            in_mean_dur,
            node["out_tcp"],
            node["out_udp"],
            node["out_icmp"],
            node["out_other_proto"],
            node["in_tcp"],
            node["in_udp"],
            node["in_icmp"],
            node["in_other_proto"],
        ]
        x.append(log1p_list(features))

    edges: List[List[int]] = []
    edge_attr: List[List[float]] = []
    for (src, dst), edge in sorted(acc.edge_stats.items()):
        n_flows = edge["n_flows"]
        mean_dur = edge["dur_sum"] / n_flows if n_flows else 0.0
        features = [
            edge["n_flows"],
            edge["sum_pkts"],
            edge["sum_bytes"],
            edge["sum_srcbytes"],
            mean_dur,
            float(len(edge["uniq_sports"])),
            float(len(edge["uniq_dports"])),
            edge["proto_tcp"],
            edge["proto_udp"],
            edge["proto_icmp"],
            edge["proto_other"],
        ]
        edges.append([ip_to_idx[src], ip_to_idx[dst]])
        edge_attr.append(log1p_list(features))

    num_edges = len(edges)
    if num_edges == 0:
        stats.graphs_skipped_no_edges += 1
        return

    window_end = window_start + timedelta(seconds=window_seconds)
    num_positive = sum(1 for value, keep in zip(labels, labeled_mask) if keep and value == 1)

    payload = {
        "scenario_name": scenario_name,
        "source_file": str(source_file),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "node_ips": node_ips,
        "x": x,
        "edges": edges,
        "edge_attr": edge_attr,
        "y": labels,
        "labeled_mask": labeled_mask,
    }

    scenario_graph_dir = graphs_dir / scenario_name
    scenario_graph_dir.mkdir(parents=True, exist_ok=True)
    stamp = window_start.strftime("%Y%m%d_%H%M%S")
    suffix = ".json.gz" if output_format == "json.gz" else ".json"
    graph_path = scenario_graph_dir / f"{scenario_name}__{stamp}{suffix}"
    write_json_payload(graph_path, payload, output_format)

    metadata_writer.write_row({
        "scenario_name": scenario_name,
        "source_file": str(source_file),
        "graph_path": str(graph_path),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_labeled_nodes": num_labeled,
        "num_positive_nodes": num_positive,
        "readme_infected_ips": len(scenario_labels.infected_ips),
        "readme_normal_ips": len(scenario_labels.normal_ips),
        "label_strategy": label_strategy,
        "output_format": output_format,
    })
    stats.graphs_saved += 1


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
    metadata_writer: MetadataWriter,
    stats: BuildStats,
    window_minutes: int,
    label_strategy: str,
    min_nodes: int,
    min_labeled_nodes: int,
    output_format: str,
    verbose: bool,
) -> None:
    scenario_name = scenario_name_for_file(binetflow_path)
    scenario_dir = scenario_dir_for_file(binetflow_path)
    scenario_labels = parse_scenario_readme(scenario_dir)
    stats.scenarios_seen += 1

    dialect = sniff_dialect(binetflow_path)
    window_seconds = window_minutes * 60
    current_window_start: Optional[datetime] = None
    current_acc = new_accumulator()
    last_timestamp: Optional[datetime] = None

    with binetflow_path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh, dialect)
        try:
            header = next(reader)
        except StopIteration:
            return

        mapping = canonicalize_header(header)

        for raw_row in reader:
            stats.rows_seen += 1
            if not raw_row or all(not str(cell).strip() for cell in raw_row):
                continue

            try:
                row = {
                    "time": parse_timestamp(raw_row[mapping["time"]]),
                    "src": str(raw_row[mapping["src"]]).strip(),
                    "dst": str(raw_row[mapping["dst"]]).strip(),
                    "proto": str(raw_row[mapping["proto"]]).strip().lower(),
                    "sport": str(raw_row[mapping["sport"]]).strip(),
                    "dport": str(raw_row[mapping["dport"]]).strip(),
                    "dur": parse_float(raw_row[mapping["dur"]]),
                    "totpkts": parse_float(raw_row[mapping["totpkts"]]),
                    "totbytes": parse_float(raw_row[mapping["totbytes"]]),
                    "srcbytes": parse_float(raw_row[mapping["srcbytes"]]),
                    "label": str(raw_row[mapping["label"]]).strip(),
                }
            except Exception:
                stats.parse_errors += 1
                continue

            if not row["src"] or not row["dst"]:
                continue

            stats.rows_kept += 1
            timestamp = row["time"]
            if last_timestamp is not None and timestamp < last_timestamp:
                stats.out_of_order_rows += 1
            last_timestamp = timestamp

            window_start = floor_window_start(timestamp, window_seconds)
            if current_window_start is None:
                current_window_start = window_start
            elif window_start != current_window_start:
                finalize_window(
                    acc=current_acc,
                    scenario_name=scenario_name,
                    source_file=binetflow_path,
                    window_start=current_window_start,
                    window_seconds=window_seconds,
                    graphs_dir=graphs_dir,
                    metadata_writer=metadata_writer,
                    scenario_labels=scenario_labels,
                    label_strategy=label_strategy,
                    min_nodes=min_nodes,
                    min_labeled_nodes=min_labeled_nodes,
                    output_format=output_format,
                    stats=stats,
                )
                current_acc = new_accumulator()
                current_window_start = window_start

            add_row_to_accumulator(current_acc, row)

    if current_window_start is not None:
        finalize_window(
            acc=current_acc,
            scenario_name=scenario_name,
            source_file=binetflow_path,
            window_start=current_window_start,
            window_seconds=window_seconds,
            graphs_dir=graphs_dir,
            metadata_writer=metadata_writer,
            scenario_labels=scenario_labels,
            label_strategy=label_strategy,
            min_nodes=min_nodes,
            min_labeled_nodes=min_labeled_nodes,
            output_format=output_format,
            stats=stats,
        )

    log(f"Processed {scenario_name}: {binetflow_path.name}", verbose)


def main() -> None:
    args = parse_args()
    graphs_dir, meta_dir = ensure_dirs(args.output_dir)
    write_feature_metadata(meta_dir)

    flow_files = discover_binetflows(args.input_root, args.file_pattern)
    if not flow_files:
        raise SystemExit(f"No flow files matching {args.file_pattern!r} found under {args.input_root}")

    stats = BuildStats()
    metadata_writer = MetadataWriter(meta_dir)

    try:
        for flow_file in flow_files:
            try:
                process_file(
                    binetflow_path=flow_file,
                    graphs_dir=graphs_dir,
                    metadata_writer=metadata_writer,
                    stats=stats,
                    window_minutes=args.window_minutes,
                    label_strategy=args.label_strategy,
                    min_nodes=args.min_nodes,
                    min_labeled_nodes=args.min_labeled_nodes,
                    output_format=args.output_format,
                    verbose=args.verbose,
                )
            except Exception as exc:
                print(f"[WARN] Failed to process {flow_file}: {exc}", flush=True)
    finally:
        metadata_writer.close()

    summary = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "window_minutes": args.window_minutes,
        "label_strategy": args.label_strategy,
        "output_format": args.output_format,
        "num_flow_files_seen": len(flow_files),
        "num_graphs_saved": stats.graphs_saved,
        "num_windows_seen": stats.windows_seen,
        "graphs_skipped_no_edges": stats.graphs_skipped_no_edges,
        "graphs_skipped_min_nodes": stats.graphs_skipped_min_nodes,
        "graphs_skipped_min_labeled": stats.graphs_skipped_min_labeled,
        "rows_seen": stats.rows_seen,
        "rows_kept": stats.rows_kept,
        "parse_errors": stats.parse_errors,
        "out_of_order_rows": stats.out_of_order_rows,
        "node_feature_dim": len(NODE_FEATURE_NAMES),
        "edge_feature_dim": len(EDGE_FEATURE_NAMES),
    }
    (meta_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote graph index to: {metadata_writer.index_path}", flush=True)


if __name__ == "__main__":
    main()
