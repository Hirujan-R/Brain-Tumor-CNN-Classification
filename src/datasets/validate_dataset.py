import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .brain_tumor_dataset import BrainTumorDataset, EXPECTED_IMAGE_SHAPE
    from .dataset_utils import summarize_registry
    from .label_mapping import VALID_ORIGINAL_LABELS, encode_label
except ImportError:
    from brain_tumor_dataset import BrainTumorDataset, EXPECTED_IMAGE_SHAPE
    from dataset_utils import summarize_registry
    from label_mapping import VALID_ORIGINAL_LABELS, encode_label


REQUIRED_COLUMNS = {
    "image_id",
    "processed_path",
    "original_path",
    "patient_id",
    "label",
    "height",
    "width",
    "channels",
    "fold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed PyTorch dataset.")
    parser.add_argument(
        "--index",
        default="data/processed/processed_index.csv",
        help="Path to processed_index.csv.",
    )
    parser.add_argument(
        "--output",
        default="reports/dataset_validation_report.json",
        help="Path to write validation report JSON.",
    )
    return parser.parse_args()


def add_issue(issues: list[dict], code: str, message: str, **details) -> None:
    issue = {"code": code, "message": message}
    issue.update(details)
    issues.append(issue)


def validate_registry(df: pd.DataFrame) -> list[dict]:
    issues = []

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        add_issue(
            issues,
            "missing_columns",
            f"Missing required columns: {sorted(missing_columns)}",
        )
        return issues

    duplicate_image_ids = df[df["image_id"].duplicated()]["image_id"].tolist()
    if duplicate_image_ids:
        add_issue(
            issues,
            "duplicate_image_ids",
            "Duplicate image IDs found.",
            examples=duplicate_image_ids[:10],
            count=len(duplicate_image_ids),
        )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        add_issue(
            issues,
            "duplicate_registry_rows",
            "Duplicate registry rows found.",
            count=duplicate_rows,
        )

    invalid_labels = sorted(set(df["label"].astype(int)) - set(VALID_ORIGINAL_LABELS))
    if invalid_labels:
        add_issue(
            issues,
            "invalid_original_labels",
            f"Invalid original labels: {invalid_labels}",
        )

    invalid_folds = sorted(set(df["fold"].astype(int)) - {0, 1, 2, 3, 4})
    if invalid_folds:
        add_issue(issues, "invalid_folds", f"Invalid fold values: {invalid_folds}")

    leaking_patients = []
    for patient_id, group in df.groupby("patient_id"):
        folds = sorted(group["fold"].unique().tolist())
        if len(folds) > 1:
            leaking_patients.append(
                {
                    "patient_id": str(patient_id),
                    "folds": [int(fold) for fold in folds],
                    "samples": int(len(group)),
                }
            )

    if leaking_patients:
        add_issue(
            issues,
            "patient_leakage",
            "Patients assigned to multiple folds.",
            examples=leaking_patients[:10],
            count=len(leaking_patients),
        )

    missing_files = [
        path for path in df["processed_path"].tolist() if not Path(path).exists()
    ]
    if missing_files:
        add_issue(
            issues,
            "missing_processed_files",
            "Processed files are missing.",
            examples=missing_files[:10],
            count=len(missing_files),
        )

    return issues


def validate_arrays(df: pd.DataFrame) -> list[dict]:
    issues = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Validating processed arrays",
    ):
        path = row["processed_path"]
        try:
            image = np.load(path, allow_pickle=False)
            if image.shape != EXPECTED_IMAGE_SHAPE:
                add_issue(
                    issues,
                    "shape_mismatch",
                    f"Expected {EXPECTED_IMAGE_SHAPE}, got {image.shape}",
                    image_id=int(row["image_id"]),
                    processed_path=path,
                )
                continue

            if image.dtype != np.float32:
                add_issue(
                    issues,
                    "dtype_mismatch",
                    f"Expected float32, got {image.dtype}",
                    image_id=int(row["image_id"]),
                    processed_path=path,
                )

            if not np.isfinite(image).all():
                add_issue(
                    issues,
                    "nonfinite_pixels",
                    "Image contains NaN or Inf.",
                    image_id=int(row["image_id"]),
                    processed_path=path,
                )

            min_value = float(image.min())
            max_value = float(image.max())
            if min_value < -1e-6 or max_value > 1.0 + 1e-6:
                add_issue(
                    issues,
                    "range_mismatch",
                    "Image values outside [0, 1].",
                    image_id=int(row["image_id"]),
                    processed_path=path,
                    min_value=min_value,
                    max_value=max_value,
                )

            if not np.allclose(image[:, :, 0], image[:, :, 1], atol=1e-6) or not np.allclose(
                image[:, :, 1], image[:, :, 2], atol=1e-6
            ):
                add_issue(
                    issues,
                    "channel_mismatch",
                    "Replicated channels are not identical.",
                    image_id=int(row["image_id"]),
                    processed_path=path,
                )

            model_label = encode_label(row["label"])
            if model_label not in {0, 1, 2}:
                add_issue(
                    issues,
                    "invalid_model_label",
                    f"Invalid encoded model label: {model_label}",
                    image_id=int(row["image_id"]),
                    original_label=int(row["label"]),
                )
        except Exception as exc:
            add_issue(
                issues,
                "array_load_error",
                str(exc),
                image_id=int(row["image_id"]),
                processed_path=path,
            )

    return issues


def validate_dataset_smoke(index_path: str) -> list[dict]:
    issues = []
    try:
        dataset = BrainTumorDataset(index_path, return_metadata=True)
        sample_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
        for index in sample_indices:
            image, label, metadata = dataset[index]
            if tuple(image.shape) != (3, 224, 224):
                add_issue(
                    issues,
                    "dataset_tensor_shape_mismatch",
                    f"Unexpected tensor shape: {tuple(image.shape)}",
                    index=index,
                )
            if int(label) not in {0, 1, 2}:
                add_issue(
                    issues,
                    "dataset_label_mismatch",
                    f"Unexpected label: {int(label)}",
                    index=index,
                )
            if not {"patient_id", "image_id", "fold", "processed_path"} <= set(metadata):
                add_issue(
                    issues,
                    "dataset_metadata_missing",
                    "Dataset metadata missing required keys.",
                    index=index,
                )
    except Exception as exc:
        add_issue(issues, "dataset_smoke_error", str(exc))

    return issues


def main() -> None:
    args = parse_args()
    start_time = datetime.now(timezone.utc)
    start_counter = perf_counter()

    df = pd.read_csv(args.index)
    issues = []
    issues.extend(validate_registry(df))

    if not issues:
        issues.extend(validate_arrays(df))
        issues.extend(validate_dataset_smoke(args.index))

    end_time = datetime.now(timezone.utc)

    report = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "runtime_seconds": perf_counter() - start_counter,
        "status": "passed" if not issues else "failed",
        "processed_index": args.index,
        "summary": summarize_registry(df) if not df.empty else {},
        "checks": {
            "missing_files": 0,
            "duplicate_image_ids": 0,
            "duplicate_registry_rows": 0,
            "invalid_labels": 0,
            "shape_mismatches": 0,
            "patient_leakage": 0,
        },
        "issues": issues,
    }

    for issue in issues:
        code = issue["code"]
        if code == "missing_processed_files":
            report["checks"]["missing_files"] = int(issue.get("count", 1))
        elif code == "duplicate_image_ids":
            report["checks"]["duplicate_image_ids"] = int(issue.get("count", 1))
        elif code == "duplicate_registry_rows":
            report["checks"]["duplicate_registry_rows"] = int(issue.get("count", 1))
        elif code in {"invalid_original_labels", "invalid_model_label"}:
            report["checks"]["invalid_labels"] += 1
        elif code == "shape_mismatch":
            report["checks"]["shape_mismatches"] += 1
        elif code == "patient_leakage":
            report["checks"]["patient_leakage"] = int(issue.get("count", 1))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=4)

    if issues:
        raise SystemExit(f"Dataset validation failed with {len(issues)} issues")

    print(f"[OK] Dataset validation passed for {len(df)} samples")


if __name__ == "__main__":
    main()
