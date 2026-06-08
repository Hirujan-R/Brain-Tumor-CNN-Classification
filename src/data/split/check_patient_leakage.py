import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

N_FOLDS = 5
SPLIT_DIR = "data/splits"
REPORT_PATH = Path(SPLIT_DIR) / "leakage_report.json"

val_dfs = []
leakage_cases = []

for fold in range(N_FOLDS):
    train = pd.read_csv(f"{SPLIT_DIR}/fold_{fold}_train.csv")
    val = pd.read_csv(f"{SPLIT_DIR}/fold_{fold}_val.csv")

    train_patients = set(train["patient_id"])
    val_patients = set(val["patient_id"])
    overlap = train_patients & val_patients

    if overlap:
        leakage_cases.extend((fold, pid) for pid in sorted(overlap))

    val = val.copy()
    val["validation_fold"] = fold
    val_dfs.append(val)

val_df = pd.concat(val_dfs, ignore_index=True)
patient_fold_counts = val_df.groupby("patient_id")["validation_fold"].nunique()
multi_val_fold_patients = patient_fold_counts[patient_fold_counts > 1]

sample_fold_counts = val_df.groupby("filepath")["validation_fold"].nunique()
multi_val_fold_samples = sample_fold_counts[sample_fold_counts > 1]

print("Patients:", val_df["patient_id"].nunique())
print("Validation samples:", len(val_df))
print("Train/val leakage cases:", len(leakage_cases))
print("Patients in multiple validation folds:", len(multi_val_fold_patients))
print("Samples in multiple validation folds:", len(multi_val_fold_samples))

if leakage_cases:
    print("Example train/val leakage:", leakage_cases[:10])

if not multi_val_fold_patients.empty:
    print("Example multi-fold patients:", multi_val_fold_patients.head(10).to_dict())

report = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "num_folds": N_FOLDS,
    "patients": int(val_df["patient_id"].nunique()),
    "validation_samples": int(len(val_df)),
    "train_val_leakage_cases": int(len(leakage_cases)),
    "patients_in_multiple_validation_folds": int(len(multi_val_fold_patients)),
    "samples_in_multiple_validation_folds": int(len(multi_val_fold_samples)),
    "status": "failed"
    if leakage_cases
    or not multi_val_fold_patients.empty
    or not multi_val_fold_samples.empty
    else "passed",
    "example_train_val_leakage": [
        {"fold": int(fold), "patient_id": str(pid)}
        for fold, pid in leakage_cases[:10]
    ],
    "example_multi_fold_patients": {
        str(pid): int(count)
        for pid, count in multi_val_fold_patients.head(10).to_dict().items()
    },
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w") as handle:
    json.dump(report, handle, indent=4)

if leakage_cases or not multi_val_fold_patients.empty or not multi_val_fold_samples.empty:
    raise SystemExit(1)
