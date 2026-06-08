import json
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    h5py = None


# -----------------------------
# CONFIG
# -----------------------------
INDEX_PATH = Path("data/ingested/index.csv")
CVIND_PATH = Path("data/splits/cvind.mat")
OUTPUT_DIR = Path("data/splits")
N_FOLDS = 5
SEED = 42


# -----------------------------
# LOAD CVIND
# -----------------------------
def load_cvind(path: Path) -> np.ndarray:
    """
    Loads cvind.mat and returns a clean 1D fold array.
    Expected shape: (3064,)
    Values: {1,2,3,4,5}
    """
    if h5py is not None:
        with h5py.File(path, "r") as f:
            cvind = np.array(f["cvind"]).squeeze().astype(int)
    else:
        print("[WARN] h5py is not installed; falling back to h5dump for cvind")
        cvind = load_cvind_with_h5dump(path)

    if cvind.ndim != 1:
        raise ValueError(f"cvind must be 1D after squeeze, got shape: {cvind.shape}")

    observed = set(cvind.tolist())
    expected = set(range(1, N_FOLDS + 1))
    if observed != expected:
        raise ValueError(f"Expected cvind values {expected}, observed {observed}")

    return cvind


def load_cvind_with_h5dump(path: Path) -> np.ndarray:
    """
    Fallback reader for local environments that have HDF5 tools but not h5py.
    The project requirements include h5py, so this is only a convenience path.
    """
    result = subprocess.run(
        ["h5dump", "-d", "/cvind", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )

    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("(0,") or ":" not in line:
            continue
        values.extend(
            int(float(value))
            for value in re.findall(r"\b[1-5](?:\.0)?\b", line.split(":", 1)[1])
        )

    return np.array(values, dtype=int)


# -----------------------------
# VALIDATION
# -----------------------------
def validate_cvind(df, cvind):
    print("[INFO] Validating official cvind consistency...")
    if len(df) != len(cvind):
        raise ValueError(
            f"cvind length mismatch: index has {len(df)} rows, "
            f"cvind has {len(cvind)}"
        )

    print("[INFO] cvind length and fold values are valid")


def sort_by_mat_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the dataset order expected by cvind.mat: 1.mat ... 3064.mat.
    """
    df = df.copy()
    df["_mat_number"] = df["filepath"].map(lambda p: int(Path(p).stem))
    df = df.sort_values("_mat_number").reset_index(drop=True)

    expected = list(range(1, len(df) + 1))
    observed = df["_mat_number"].tolist()
    if observed != expected:
        raise ValueError(
            "Cannot align cvind safely: expected contiguous numeric filenames "
            f"1..{len(df)}"
        )

    return df


def assign_official_folds(df: pd.DataFrame, cvind: np.ndarray) -> pd.DataFrame:
    """
    Assign folds from the paper-provided cvind.mat.

    cvind uses MATLAB-style fold labels 1..5. The exported files use Python-style
    names fold_0..fold_4, so we store folds internally as 0..4.
    """
    df = sort_by_mat_number(df)
    validate_cvind(df, cvind)

    df["fold"] = cvind - 1
    df = df.drop(columns=["_mat_number"])
    return df


# -----------------------------
# LEAKAGE CHECK (POST-SPLIT)
# -----------------------------
def validate_no_leakage(df: pd.DataFrame, n_folds: int):
    leakage_cases = 0

    for pid, g in df.groupby("patient_id"):
        if g["fold"].nunique() > 1:
            leakage_cases += 1

    if leakage_cases > 0:
        raise RuntimeError(f"LEAKAGE DETECTED: {leakage_cases} patients")

    observed_folds = sorted(df["fold"].unique().tolist())
    expected_folds = list(range(n_folds))
    if observed_folds != expected_folds:
        raise RuntimeError(
            f"Expected folds {expected_folds}, observed {observed_folds}"
        )

    print("[OK] No patient leakage detected")
    return leakage_cases


# -----------------------------
# EXPORT
# -----------------------------
def export_splits(df: pd.DataFrame, out_dir: Path, n_folds: int):
    """
    Writes fold-wise train/val CSVs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_stats = {}

    for fold in range(n_folds):

        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]

        train_path = out_dir / f"fold_{fold}_train.csv"
        val_path = out_dir / f"fold_{fold}_val.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        fold_stats[f"fold_{fold}"] = {
            "train_patients": int(train_df["patient_id"].nunique()),
            "val_patients": int(val_df["patient_id"].nunique()),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "train_class_distribution": {
                str(k): int(v)
                for k, v in train_df["label"].value_counts().sort_index().items()
            },
            "val_class_distribution": {
                str(k): int(v)
                for k, v in val_df["label"].value_counts().sort_index().items()
            },
            "train_patient_class_distribution": {
                str(k): int(v)
                for k, v in (
                    train_df.drop_duplicates("patient_id")["label"]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
            "val_patient_class_distribution": {
                str(k): int(v)
                for k, v in (
                    val_df.drop_duplicates("patient_id")["label"]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
        }

    return fold_stats


# -----------------------------
# REPORT
# -----------------------------
def write_report(
    df: pd.DataFrame,
    leakage_count: int,
    fold_stats: dict,
    patient_map: dict,
    n_folds: int,
    seed: int,
):
    """
    Writes final audit report.
    """
    report = {
        "num_samples": len(df),
        "num_patients": df["patient_id"].nunique(),
        "num_folds": n_folds,
        "split_unit": "patient",
        "fold_source": "data/splits/cvind.mat",
        "fold_alignment": "cvind[i] maps to numeric MATLAB file i+1, after sorting 1.mat..N.mat",
        "cvind_usage": "official_fold_assignment",
        "patient_leakage_cases": leakage_count,
        "patient_fold_map": patient_map,
        "fold_distribution": {
            k: v for k, v in fold_stats.items()
        },
    }

    with open(OUTPUT_DIR / "split_report.json", "w") as f:
        json.dump(report, f, indent=4)


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("\n[CV SPLIT] Starting official paper-fold pipeline...\n")

    df = pd.read_csv(INDEX_PATH)

    print(f"[INFO] Samples: {len(df)}")
    print(f"[INFO] Patients: {df['patient_id'].nunique()}")

    cvind = load_cvind(CVIND_PATH)
    df = assign_official_folds(df, cvind)
    patient_map = {
        str(pid): int(group["fold"].iloc[0])
        for pid, group in df.groupby("patient_id")
    }

    print("[INFO] Official cvind fold assignment complete")

    leakage_count = validate_no_leakage(df, n_folds=N_FOLDS)

    fold_stats = export_splits(df, OUTPUT_DIR, n_folds=N_FOLDS)
    write_report(
        df=df,
        leakage_count=leakage_count,
        fold_stats=fold_stats,
        patient_map=patient_map,
        n_folds=N_FOLDS,
        seed=SEED,
    )

    print("\n[CV SPLIT] COMPLETED (OFFICIAL PAPER FOLDS)\n")


if __name__ == "__main__":

    main()
