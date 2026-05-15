import csv
from collections import defaultdict

INDEX_CSV = "./ctu13_pyg/metadata/graphs_index_pyg.csv"

FAMILY_MAP = {
    "CTU-Malware-Capture-Botnet-42": "Neris",
    "CTU-Malware-Capture-Botnet-43": "Neris",
    "CTU-Malware-Capture-Botnet-50": "Neris",
    "CTU-Malware-Capture-Botnet-44": "Rbot",
    "CTU-Malware-Capture-Botnet-45": "Rbot",
    "CTU-Malware-Capture-Botnet-51": "Rbot",
    "CTU-Malware-Capture-Botnet-52": "Rbot",
    "CTU-Malware-Capture-Botnet-46": "Virut",
    "CTU-Malware-Capture-Botnet-54": "Virut",
    "CTU-Malware-Capture-Botnet-47": "DonBot",
    "CTU-Malware-Capture-Botnet-48": "Sogou",
    "CTU-Malware-Capture-Botnet-49": "Murlo",
    "CTU-Malware-Capture-Botnet-53": "NSIS.ay",
}

def load_rows(index_csv):
    with open(index_csv, "r", newline="") as f:
        return list(csv.DictReader(f))

def build_family_to_rows(rows):
    family_to_rows = defaultdict(list)
    for row in rows:
        scenario = row["scenario_name"]
        family = FAMILY_MAP[scenario]
        family_to_rows[family].append(row)
    return family_to_rows

def make_logo_folds(rows):
    family_to_rows = build_family_to_rows(rows)
    families = sorted(family_to_rows.keys())
    folds = []

    for test_family in families:
        remaining = [f for f in families if f != test_family]

        for val_family in remaining:
            train_families = [f for f in families if f not in {test_family, val_family}]

            train_rows = []
            val_rows = []
            test_rows = []

            for fam in train_families:
                train_rows.extend(family_to_rows[fam])
            val_rows.extend(family_to_rows[val_family])
            test_rows.extend(family_to_rows[test_family])

            folds.append({
                "train_families": train_families,
                "val_family": val_family,
                "test_family": test_family,
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
            })

    return folds

rows = load_rows(INDEX_CSV)
folds = make_logo_folds(rows)

print(f"Built {len(folds)} folds.")
for i, fold in enumerate(folds[:5], 1):
    print(f"Fold {i}")
    print("  train:", fold["train_families"])
    print("  val  :", fold["val_family"])
    print("  test :", fold["test_family"])
    print("  counts:",
          len(fold["train_rows"]),
          len(fold["val_rows"]),
          len(fold["test_rows"]))
