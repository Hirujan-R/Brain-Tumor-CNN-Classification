import pandas as pd

N_FOLDS = 5
SPLIT_DIR = "data/splits"

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

if leakage_cases or not multi_val_fold_patients.empty or not multi_val_fold_samples.empty:
    raise SystemExit(1)
