#!/usr/bin/env python3
"""
Search family-disjoint train/val/test splits for CTU-13 graphs.

This script reads a graphs_index_pyg.csv-style manifest, counts graphs per
scenario, groups scenarios by malware family, enumerates all family-disjoint
assignments to train/val/test, and ranks them by how close they are to a target
ratio such as 70/15/15.

It can also write split_train.csv, split_val.csv, split_test.csv, and a
splits.json file for the best assignment.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_FAMILY_MAP = {
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

SPLITS = ("train", "val", "test")


def parse_ratio(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Ratio must have exactly 3 comma-separated values, e.g. 70,15,15"
        )
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Ratio values must be numeric, e.g. 70,15,15"
        ) from exc
    total = sum(values)
    if total <= 0:
        raise argparse.ArgumentTypeError("Ratio total must be > 0")
    return (values[0] / total, values[1] / total, values[2] / total)


def load_family_map(path: Path | None) -> Dict[str, str]:
    if path is None:
        return dict(DEFAULT_FAMILY_MAP)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Family map JSON must be an object: {scenario_name: family}")
    return {str(k): str(v) for k, v in data.items()}


def load_manifest_rows(index_csv: Path) -> Tuple[List[dict], List[str]]:
    with index_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise RuntimeError(f"Manifest has no header: {index_csv}")
    required = {"scenario_name"}
    missing = required - set(fieldnames)
    if missing:
        raise RuntimeError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )
    return rows, fieldnames


def count_graphs_per_scenario(rows: Sequence[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[str(row["scenario_name"])] += 1
    return counts


def build_family_summary(
    scenario_counts: Counter,
    family_map: Dict[str, str],
) -> Tuple[Dict[str, int], Dict[str, List[str]], List[str]]:
    unknown_scenarios = [
        scenario for scenario in sorted(scenario_counts) if scenario not in family_map
    ]
    if unknown_scenarios:
        raise RuntimeError(
            "No family mapping available for scenario(s): "
            + ", ".join(unknown_scenarios)
        )

    family_counts: Dict[str, int] = defaultdict(int)
    family_to_scenarios: Dict[str, List[str]] = defaultdict(list)
    for scenario, count in scenario_counts.items():
        family = family_map[scenario]
        family_counts[family] += count
        family_to_scenarios[family].append(scenario)

    for family in family_to_scenarios:
        family_to_scenarios[family].sort()

    families = sorted(family_to_scenarios)
    return dict(family_counts), dict(family_to_scenarios), families


def assignment_is_valid(
    assignment: Dict[str, str],
    family_to_scenarios: Dict[str, List[str]],
    min_families: Dict[str, int],
    min_scenarios: Dict[str, int],
) -> bool:
    families_per_split = Counter(assignment.values())
    scenarios_per_split = Counter()
    for family, split in assignment.items():
        scenarios_per_split[split] += len(family_to_scenarios[family])

    for split in SPLITS:
        if families_per_split[split] < min_families[split]:
            return False
        if scenarios_per_split[split] < min_scenarios[split]:
            return False
    return True


def score_assignment(
    split_counts: Dict[str, int],
    total_graphs: int,
    target_ratio: Tuple[float, float, float],
) -> float:
    ratios = {
        "train": split_counts["train"] / total_graphs,
        "val": split_counts["val"] / total_graphs,
        "test": split_counts["test"] / total_graphs,
    }

    # Weighted squared error with a slight extra penalty if train undershoots its target.
    err = 0.0
    for split, target in zip(SPLITS, target_ratio):
        diff = ratios[split] - target
        weight = 1.0
        if split == "train" and diff < 0:
            weight = 1.35
        err += weight * (diff * diff)
    return err


def enumerate_assignments(
    families: Sequence[str],
    family_counts: Dict[str, int],
    family_to_scenarios: Dict[str, List[str]],
    target_ratio: Tuple[float, float, float],
    min_families: Dict[str, int],
    min_scenarios: Dict[str, int],
) -> List[dict]:
    total_graphs = sum(family_counts.values())
    candidates: List[dict] = []

    for choices in itertools.product(SPLITS, repeat=len(families)):
        assignment = dict(zip(families, choices))
        if not assignment_is_valid(
            assignment=assignment,
            family_to_scenarios=family_to_scenarios,
            min_families=min_families,
            min_scenarios=min_scenarios,
        ):
            continue

        split_counts = {"train": 0, "val": 0, "test": 0}
        split_families = {"train": [], "val": [], "test": []}
        split_scenarios = {"train": [], "val": [], "test": []}

        for family, split in assignment.items():
            split_counts[split] += family_counts[family]
            split_families[split].append(family)
            split_scenarios[split].extend(family_to_scenarios[family])

        score = score_assignment(split_counts, total_graphs, target_ratio)
        split_ratios = {
            split: split_counts[split] / total_graphs for split in SPLITS
        }

        for split in SPLITS:
            split_families[split].sort()
            split_scenarios[split].sort()

        candidates.append(
            {
                "score": score,
                "counts": split_counts,
                "ratios": split_ratios,
                "families": split_families,
                "scenarios": split_scenarios,
            }
        )

    candidates.sort(
        key=lambda c: (
            c["score"],
            abs(c["counts"]["train"] - c["counts"]["val"]),
            -c["counts"]["train"],
        )
    )
    return candidates


def print_summary(
    scenario_counts: Counter,
    family_counts: Dict[str, int],
    family_to_scenarios: Dict[str, List[str]],
) -> None:
    print("Graphs per scenario:")
    for scenario in sorted(scenario_counts):
        family = family_to_scenarios_lookup(scenario, family_to_scenarios)
        print(f"  {scenario}: {scenario_counts[scenario]}  ({family})")

    print("\nGraphs per family:")
    for family in sorted(family_counts):
        scenarios = ", ".join(family_to_scenarios[family])
        print(f"  {family}: {family_counts[family]}  [{scenarios}]")


def family_to_scenarios_lookup(
    scenario: str,
    family_to_scenarios: Dict[str, List[str]],
) -> str:
    for family, scenarios in family_to_scenarios.items():
        if scenario in scenarios:
            return family
    return "UNKNOWN"


def print_candidate(rank: int, candidate: dict) -> None:
    print(f"\nCandidate #{rank}")
    print(f"  score: {candidate['score']:.8f}")
    for split in SPLITS:
        count = candidate["counts"][split]
        ratio = candidate["ratios"][split] * 100.0
        families = ", ".join(candidate["families"][split])
        scenarios = ", ".join(candidate["scenarios"][split])
        print(f"  {split:>5}: {count:>4} graphs ({ratio:5.1f}%)")
        print(f"         families:  {families}")
        print(f"         scenarios: {scenarios}")


def write_split_files(
    output_dir: Path,
    rows: Sequence[dict],
    fieldnames: Sequence[str],
    best_candidate: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_to_split: Dict[str, str] = {}
    for split in SPLITS:
        for scenario in best_candidate["scenarios"][split]:
            scenario_to_split[scenario] = split

    out_fieldnames = list(fieldnames)
    if "split" not in out_fieldnames:
        out_fieldnames.append("split")

    split_rows: Dict[str, List[dict]] = {split: [] for split in SPLITS}
    for row in rows:
        scenario = str(row["scenario_name"])
        split = scenario_to_split[scenario]
        out_row = dict(row)
        out_row["split"] = split
        split_rows[split].append(out_row)

    for split in SPLITS:
        out_path = output_dir / f"split_{split}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(split_rows[split])

    split_json = {
        split: best_candidate["scenarios"][split] for split in SPLITS
    }
    with (output_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=2)

    details = {
        "score": best_candidate["score"],
        "counts": best_candidate["counts"],
        "ratios": best_candidate["ratios"],
        "families": best_candidate["families"],
        "scenarios": best_candidate["scenarios"],
    }
    with (output_dir / "best_split_details.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search family-disjoint CTU-13 train/val/test splits and rank them by "
            "closeness to a target graph-count ratio."
        )
    )
    parser.add_argument(
        "--index-csv",
        required=True,
        type=Path,
        help="Path to graphs_index_pyg.csv",
    )
    parser.add_argument(
        "--family-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping scenario_name to family. "
            "If omitted, the built-in CTU-13 map is used."
        ),
    )
    parser.add_argument(
        "--target-ratio",
        type=parse_ratio,
        default=parse_ratio("70,15,15"),
        help="Desired train,val,test ratio as percentages or weights, e.g. 70,15,15",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top candidate splits to print",
    )
    parser.add_argument(
        "--min-train-families",
        type=int,
        default=3,
        help="Minimum number of malware families required in train",
    )
    parser.add_argument(
        "--min-val-families",
        type=int,
        default=1,
        help="Minimum number of malware families required in val",
    )
    parser.add_argument(
        "--min-test-families",
        type=int,
        default=1,
        help="Minimum number of malware families required in test",
    )
    parser.add_argument(
        "--min-train-scenarios",
        type=int,
        default=4,
        help="Minimum number of scenarios required in train",
    )
    parser.add_argument(
        "--min-val-scenarios",
        type=int,
        default=1,
        help="Minimum number of scenarios required in val",
    )
    parser.add_argument(
        "--min-test-scenarios",
        type=int,
        default=1,
        help="Minimum number of scenarios required in test",
    )
    parser.add_argument(
        "--write-best-split",
        action="store_true",
        help="Write split_train.csv, split_val.csv, split_test.csv, and JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write split CSV/JSON files into when --write-best-split is used. "
            "Defaults to the directory containing --index-csv."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print scenario/family count summaries before the ranked candidates",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rows, fieldnames = load_manifest_rows(args.index_csv)
    family_map = load_family_map(args.family_map)
    scenario_counts = count_graphs_per_scenario(rows)
    family_counts, family_to_scenarios, families = build_family_summary(
        scenario_counts=scenario_counts,
        family_map=family_map,
    )

    if not args.quiet:
        print_summary(
            scenario_counts=scenario_counts,
            family_counts=family_counts,
            family_to_scenarios=family_to_scenarios,
        )
        print()

    min_families = {
        "train": args.min_train_families,
        "val": args.min_val_families,
        "test": args.min_test_families,
    }
    min_scenarios = {
        "train": args.min_train_scenarios,
        "val": args.min_val_scenarios,
        "test": args.min_test_scenarios,
    }

    candidates = enumerate_assignments(
        families=families,
        family_counts=family_counts,
        family_to_scenarios=family_to_scenarios,
        target_ratio=args.target_ratio,
        min_families=min_families,
        min_scenarios=min_scenarios,
    )

    if not candidates:
        raise RuntimeError(
            "No valid family-disjoint split assignments matched the requested constraints. "
            "Try lowering the minimum family/scenario requirements."
        )

    limit = max(1, args.top_k)
    for idx, candidate in enumerate(candidates[:limit], start=1):
        print_candidate(idx, candidate)

    if args.write_best_split:
        output_dir = args.output_dir or args.index_csv.parent
        write_split_files(
            output_dir=output_dir,
            rows=rows,
            fieldnames=fieldnames,
            best_candidate=candidates[0],
        )
        print(f"\nWrote best split files to: {output_dir}")


if __name__ == "__main__":
    main()
