# CTU-13 Data Preprocessing

This directory contains a CTU-13 botnet-flow preprocessing pipeline for graph neural network experiments. It includes selected CTU-13 bidirectional flow files grouped by malware family, plus scripts for building time-window communication graphs, converting them to PyTorch Geometric format, and creating train/validation/test splits.

## Contents

```text
.
├── data/
│   ├── donbotFamily/
│   ├── murloFamily/
│   ├── nerisFamily/
│   ├── nsisFamily/
│   ├── rbotFamily/
│   ├── sogouFamily/
│   └── virutFamily/
├── ctu13_graph_builder.py
├── ctu13_graph_builder_stdlib.py
├── ctu13_json_to_pyg.py
├── foldBuilder.py
├── make_splits.py
└── search_ctu13_splits.py
```

Each scenario directory contains:

- `README.md`: scenario metadata from CTU-13, including malware family, duration, infected hosts, normal hosts, and labeling notes.
- `*.binetflow`: bidirectional NetFlow-style CSV data used by the preprocessing scripts.

The raw `.pcap` files referenced in the CTU-13 scenario READMEs are not included in this archive.

## Dataset inventory

Rows below are `.binetflow` records, excluding the header row. File sizes are extracted-file sizes.

| CTU-13 scenario | Directory | Malware family | Flow file | Rows | Size | Duration | Infected hosts |
|---:|---|---|---|---:|---:|---|---|
| 1 | `CTU-Malware-Capture-Botnet-42` | Neris | `capture20110810.binetflow` | 2,824,636 | 368.7 MB | 6.15 hours | `147.32.84.165` |
| 2 | `CTU-Malware-Capture-Botnet-43` | Neris | `capture20110811.binetflow` | 1,808,122 | 235.8 MB | 3 hours, 51 minutes | `147.32.84.165` |
| 3 | `CTU-Malware-Capture-Botnet-44` | Rbot | `capture20110812.binetflow` | 4,710,638 | 610.0 MB | 2 days, 18 hours, 49 minutes | `147.32.84.165` |
| 4 | `CTU-Malware-Capture-Botnet-45` | Rbot | `capture20110815.binetflow` | 1,121,076 | 146.7 MB | 4 hours, 29 minutes | `147.32.84.165` |
| 5 | `CTU-Malware-Capture-Botnet-46` | Virut | `capture20110815-2.binetflow` | 129,832 | 16.9 MB | 30 minutes | `147.32.84.165` |
| 6 | `CTU-Malware-Capture-Botnet-47` | DonBot | `capture20110816.binetflow` | 418,787 | 54.9 MB | 2 hours, 9 minutes | `147.32.84.165` |
| 7 | `CTU-Malware-Capture-Botnet-48` | Sogou | `capture20110816-2.binetflow` | 114,077 | 14.9 MB | 21 minutes | `147.32.84.165` |
| 8 | `CTU-Malware-Capture-Botnet-49` | Murlo | `capture20110816-3.binetflow` | 2,954,230 | 385.2 MB | 19 hours, 29 minutes | `147.32.84.165` |
| 9 | `CTU-Malware-Capture-Botnet-50` | Neris | `capture20110817.binetflow` | 2,087,508 | 272.6 MB | 5 hours, 38 minutes | `147.32.84.165`, `147.32.84.191`, `147.32.84.192`, `147.32.84.193`, `147.32.84.204`, `147.32.84.205`, `147.32.84.206`, `147.32.84.207`, `147.32.84.208`, `147.32.84.209` |
| 10 | `CTU-Malware-Capture-Botnet-51` | Rbot | `capture20110818.binetflow` | 1,309,791 | 170.8 MB | 5 hours, 8 minutes | `147.32.84.165`, `147.32.84.191`, `147.32.84.192`, `147.32.84.193`, `147.32.84.204`, `147.32.84.205`, `147.32.84.206`, `147.32.84.207`, `147.32.84.208`, `147.32.84.209` |
| 11 | `CTU-Malware-Capture-Botnet-52` | Rbot | `capture20110818-2.binetflow` | 107,251 | 13.9 MB | 16 minutes | `147.32.84.165`, `147.32.84.191`, `147.32.84.192` |
| 12 | `CTU-Malware-Capture-Botnet-53` | NSIS.ay | `capture20110819.binetflow` | 325,471 | 42.6 MB | 1 hour, 43 minutes | `147.32.84.165`, `147.32.84.191`, `147.32.84.192` |
| 13 | `CTU-Malware-Capture-Botnet-54` | Virut | `capture20110815-3.binetflow` | 1,925,149 | 250.5 MB | 16 hours, 23 minutes | `147.32.84.165` |

## Input data format

The `.binetflow` files are CSV files with columns similar to:

```text
StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label
```

The graph builders normalize column names internally, but they require timestamp, source IP, destination IP, protocol, source port, destination port, duration, packet count, byte count, source-byte count, and label fields.

## Labeling convention

The scripts produce node labels for botnet detection:

| Label | Meaning |
|---:|---|
| `1` | Botnet / infected |
| `0` | Normal |
| `-1` | Ignore / unlabeled |

Important CTU-13 label semantics:

- Flow labels containing `From-Botnet` are treated as malicious source activity.
- Flow labels containing `From-Normal` are treated as normal source activity.
- `To-Botnet` and `To-Normal` describe traffic sent to those hosts and should not automatically be interpreted as malicious or normal source behavior.
- With the default `hybrid` strategy, the builders first use infected/normal IPs parsed from each scenario README and then fall back to flow-level labels when README labels are unavailable for a node.

## Scripts

### `ctu13_graph_builder.py`

Builds PyTorch Geometric `.pt` graph snapshots directly from `.binetflow` files. This version uses `pandas`, `numpy`, `torch`, and `torch-geometric`, and may require substantial memory for the largest flow files.

For each fixed time window, it creates a directed communication graph:

- nodes are IP addresses;
- edges are directed source-to-destination host pairs;
- node features summarize inbound/outbound degree, flow count, byte count, packet count, peer diversity, port diversity, duration, and protocol counts;
- edge features summarize flow count, packets, bytes, source bytes, duration, unique ports, and protocol counts;
- numeric features are transformed with `log1p`.

Main outputs:

```text
<output-dir>/
├── graphs/<scenario>/*.pt
└── metadata/
    ├── graphs_index.csv
    ├── feature_names.json
    └── build_summary.json
```

Example:

```bash
python3 ctu13_graph_builder.py \
  --input-root ./data \
  --output-dir ./ctu13_pyg_direct \
  --window-minutes 5 \
  --label-strategy hybrid \
  --verbose
```

Useful options:

```text
--window-minutes       Window size in minutes; default: 5
--label-strategy       readme, flow, or hybrid; default: hybrid
--min-nodes            Skip graphs with fewer nodes; default: 2
--min-labeled-nodes    Skip graphs with fewer labeled nodes; default: 1
--file-pattern         Flow-file glob; default: *.binetflow
```

### `ctu13_graph_builder_stdlib.py`

Memory-conscious graph builder that uses only Python's standard library. It streams `.binetflow` files and writes graph snapshots as JSON or compressed JSON. Use this when the direct PyG builder is too memory-heavy.

Main outputs:

```text
<output-dir>/
├── graphs/<scenario>/*.json.gz   # default
└── metadata/
    ├── graphs_index.csv
    ├── feature_names.json
    └── build_summary.json
```

Example:

```bash
python3 ctu13_graph_builder_stdlib.py \
  --input-root ./data \
  --output-dir ./ctu13_json_graphs \
  --window-minutes 5 \
  --label-strategy hybrid \
  --output-format json.gz \
  --verbose
```

### `ctu13_json_to_pyg.py`

Converts the JSON/JSON.GZ graph snapshots produced by `ctu13_graph_builder_stdlib.py` into PyTorch Geometric `.pt` files.

Main outputs:

```text
<output-dir>/
├── graphs/<scenario>/*.pt
└── metadata/
    ├── graphs_index_pyg.csv
    ├── feature_names.json
    └── conversion_summary.json
```

Example:

```bash
python3 ctu13_json_to_pyg.py \
  --input-root ./ctu13_json_graphs \
  --output-dir ./ctu13_pyg \
  --verbose
```

Useful options:

```text
--glob                 Additional graph filename glob; default: *.json.gz and *.json
--overwrite            Replace existing .pt files
--save-manifest-only   Write metadata without saving .pt graph files
```

### `search_ctu13_splits.py`

Searches for family-disjoint train/validation/test splits. It reads a PyG graph manifest, groups scenarios by malware family, enumerates split assignments, and ranks them by closeness to a target graph-count ratio.

Example:

```bash
python3 search_ctu13_splits.py \
  --index-csv ./ctu13_pyg/metadata/graphs_index_pyg.csv \
  --target-ratio 70,15,15 \
  --top-k 10 \
  --write-best-split
```

When `--write-best-split` is used, it writes:

```text
split_train.csv
split_val.csv
split_test.csv
splits.json
best_split_details.json
```

### `make_splits.py`

Creates a fixed scenario-level split from `./ctu13_pyg/metadata/graphs_index_pyg.csv`.

Built-in split:

- Train: scenarios 42, 43, 47, 49, 50, 53
- Validation: scenarios 46, 54
- Test: scenarios 44, 45, 48, 51, 52

Run from the repository root after creating `./ctu13_pyg/metadata/graphs_index_pyg.csv`:

```bash
python3 make_splits.py
```

Outputs are written beside the manifest:

```text
./ctu13_pyg/metadata/split_train.csv
./ctu13_pyg/metadata/split_val.csv
./ctu13_pyg/metadata/split_test.csv
./ctu13_pyg/metadata/splits.json
```

Note: this script uses a hard-coded manifest path. Edit `INDEX_CSV` if your PyG output directory is different.

### `foldBuilder.py`

Builds leave-one-family-out style fold definitions from `./ctu13_pyg/metadata/graphs_index_pyg.csv`. For each fold, one malware family is held out for test, one remaining family is used for validation, and the rest are used for training.

Run:

```bash
python3 foldBuilder.py
```

The current script prints fold summaries but does not write fold CSVs. Like `make_splits.py`, it uses a hard-coded `INDEX_CSV`; edit that value if needed.

## Recommended workflows

### Option A: Direct PyG build

Use this when you have enough memory for large `.binetflow` files.

```bash
python3 ctu13_graph_builder.py \
  --input-root ./data \
  --output-dir ./ctu13_pyg \
  --window-minutes 5 \
  --label-strategy hybrid \
  --verbose
```

Then create splits:

```bash
python3 search_ctu13_splits.py \
  --index-csv ./ctu13_pyg/metadata/graphs_index.csv \
  --target-ratio 70,15,15 \
  --write-best-split
```

### Option B: Streaming JSON build, then PyG conversion

Use this for lower memory usage and better resumability.

```bash
python3 ctu13_graph_builder_stdlib.py \
  --input-root ./data \
  --output-dir ./ctu13_json_graphs \
  --window-minutes 5 \
  --label-strategy hybrid \
  --output-format json.gz \
  --verbose

python3 ctu13_json_to_pyg.py \
  --input-root ./ctu13_json_graphs \
  --output-dir ./ctu13_pyg \
  --verbose

python3 search_ctu13_splits.py \
  --index-csv ./ctu13_pyg/metadata/graphs_index_pyg.csv \
  --target-ratio 70,15,15 \
  --write-best-split
```

## Output graph schema

PyTorch Geometric `Data` objects contain:

```text
x             [num_nodes, 24]    float32 node feature matrix
edge_index    [2, num_edges]     int64 directed edges
edge_attr     [num_edges, 11]    float32 edge feature matrix
y             [num_nodes]        int64 node labels: 1, 0, or -1
labeled_mask  [num_nodes]        bool mask for nodes with usable labels
```

Additional metadata is attached to graph objects, depending on the path used:

```text
scenario_name
source_file
window_start
window_end
node_id or node_ips
node_feature_names
edge_feature_names
```

The direct builder also writes `.nodes.json` sidecar files next to `.pt` graphs so node indices can be mapped back to IP addresses.

## Feature names

Node features, in order:

```text
out_degree, in_degree,
out_flows, in_flows,
out_bytes, in_bytes,
out_pkts, in_pkts,
out_srcbytes, in_srcbytes,
out_unique_peers, in_unique_peers,
out_unique_dports, in_unique_sports,
out_mean_dur, in_mean_dur,
out_tcp, out_udp, out_icmp, out_other_proto,
in_tcp, in_udp, in_icmp, in_other_proto
```

Edge features, in order:

```text
n_flows, sum_pkts, sum_bytes, sum_srcbytes, mean_dur,
uniq_sports, uniq_dports,
proto_tcp, proto_udp, proto_icmp, proto_other
```

The same information is written to `metadata/feature_names.json`.

## Environment

Recommended Python version: Python 3.10 or newer.

For the direct PyG path:

```bash
pip install numpy pandas torch torch-geometric
```

For the streaming JSON builder only:

```bash
# No third-party Python packages required.
python3 ctu13_graph_builder_stdlib.py --help
```

For JSON-to-PyG conversion:

```bash
pip install torch torch-geometric
```

`torch` and `torch-geometric` installation commands may vary by operating system, CUDA version, and package index. Use the installation instructions that match your runtime.

## Reproducibility notes

- Split by scenario or malware family, not by individual graph windows, to avoid temporal and scenario leakage.
- Fit normalization or standardization parameters on training graphs only. The builders already apply `log1p`, but they do not z-score features.
- Keep `metadata/graphs_index.csv` or `metadata/graphs_index_pyg.csv` with experiments; these files define which graph snapshot came from which scenario, source file, and time window.
- The largest flow files are several hundred MB each. Prefer `ctu13_graph_builder_stdlib.py` when working on memory-constrained machines.

## Attribution

The scenario READMEs state that the data was generated in the Stratosphere Lab as part of the Malware Capture Facility Project at CTU University, Prague. Cite the CTU-13 / Stratosphere Malware Capture Facility dataset when using these files in research. 

This README was generated by ChatGPT.
