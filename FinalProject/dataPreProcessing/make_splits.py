import csv
import json
from pathlib import Path

INDEX_CSV = Path("./ctu13_pyg/metadata/graphs_index_pyg.csv")
OUT_DIR = INDEX_CSV.parent

TRAIN_SCENARIOS = {
    "CTU-Malware-Capture-Botnet-42",
    "CTU-Malware-Capture-Botnet-43",
    "CTU-Malware-Capture-Botnet-47",
    "CTU-Malware-Capture-Botnet-49",
    "CTU-Malware-Capture-Botnet-50",
    "CTU-Malware-Capture-Botnet-53",
}

VAL_SCENARIOS = {
    "CTU-Malware-Capture-Botnet-46",
    "CTU-Malware-Capture-Botnet-54",
}

TEST_SCENARIOS = {
    "CTU-Malware-Capture-Botnet-44",
    "CTU-Malware-Capture-Botnet-45",
    "CTU-Malware-Capture-Botnet-48",
    "CTU-Malware-Capture-Botnet-51",
    "CTU-Malware-Capture-Botnet-52",
}

ALL_SPLIT_SCENARIOS = TRAIN_SCENARIOS | VAL_SCENARIOS | TEST_SCENARIOS

def get_split(scenario_name: str) -> str:
    if scenario_name in TRAIN_SCENARIOS:
        return "train"
    if scenario_name in VAL_SCENARIOS:
        return "val"
    if scenario_name in TEST_SCENARIOS:
        return "test"
    raise ValueError(f"Scenario not assigned to any split: {scenario_name}")

def main():
    with INDEX_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise RuntimeError("graphs_index_pyg.csv has no header")

    split_rows = {"train": [], "val": [], "test": []}
    seen_scenarios = set()

    for row in rows:
        scenario = row["scenario_name"]
        seen_scenarios.add(scenario)
        split = get_split(scenario)
        row = dict(row)
        row["split"] = split
        split_rows[split].append(row)

    missing = ALL_SPLIT_SCENARIOS - seen_scenarios
    if missing:
        print("Warning: these assigned scenarios were not found in the manifest:")
        for s in sorted(missing):
            print("  ", s)

    out_fieldnames = list(fieldnames)
    if "split" not in out_fieldnames:
        out_fieldnames.append("split")

    for split_name, rows_for_split in split_rows.items():
        out_csv = OUT_DIR / f"split_{split_name}.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_split)
        print(f"Wrote {out_csv} with {len(rows_for_split)} graphs")

    split_json = {
        "train": sorted(TRAIN_SCENARIOS),
        "val": sorted(VAL_SCENARIOS),
        "test": sorted(TEST_SCENARIOS),
    }
    with (OUT_DIR / "splits.json").open("w") as f:
        json.dump(split_json, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
