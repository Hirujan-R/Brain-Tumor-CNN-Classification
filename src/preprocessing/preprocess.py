from datetime import datetime, timezone
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from config import (
    CHANNELS,
    CLEAN_OUTPUT,
    EXPECTED_SOURCE_SHAPE,
    IMAGES_DIR,
    INDEX_PATH,
    INTERPOLATION,
    N_FOLDS,
    NORMALIZATION,
    OUTPUT_DIR,
    OUTPUT_DTYPE,
    PREPROCESSING_REPORT_PATH,
    PREPROCESSING_STATS_PATH,
    PROCESSED_INDEX_PATH,
    SPLIT_DIR,
    TARGET_HEIGHT,
    TARGET_SHAPE,
    TARGET_WIDTH,
)
from image_loader import load_mat_image
from registry import attach_folds, build_fold_map, load_index
from report import write_json
from save_processed import prepare_output_dirs, processed_path, save_processed_image
from transforms import minmax_normalize, replicate_channels, resize_image
from validation import (
    validate_normalized_image,
    validate_processed_image,
    validate_raw_image,
)


def build_processed_record(row: pd.Series, output_path) -> dict:
    return {
        "image_id": int(row["image_id"]),
        "processed_path": output_path.as_posix(),
        "original_path": row["filepath"],
        "patient_id": row["patient_id"],
        "label": int(row["label"]),
        "height": TARGET_HEIGHT,
        "width": TARGET_WIDTH,
        "channels": CHANNELS,
        "fold": int(row["fold"]),
    }


def main() -> None:
    start_time = datetime.now(timezone.utc)
    start_counter = perf_counter()

    validation_errors = []
    validation_warnings = []
    records = []

    index_df = load_index(INDEX_PATH)
    fold_map = build_fold_map(SPLIT_DIR, N_FOLDS)
    registry_df = attach_folds(index_df, fold_map)

    prepare_output_dirs(OUTPUT_DIR, IMAGES_DIR, clean=CLEAN_OUTPUT)

    for _, row in tqdm(
        registry_df.iterrows(),
        total=len(registry_df),
        desc="Preprocessing MRI images",
    ):
        filepath = row["filepath"]
        output_path = processed_path(IMAGES_DIR, int(row["image_id"]))

        try:
            image = load_mat_image(filepath)
            validation_warnings.extend(
                validate_raw_image(image, filepath, EXPECTED_SOURCE_SHAPE)
            )

            normalized = minmax_normalize(image)
            validate_normalized_image(normalized, filepath)

            resized = resize_image(
                normalized,
                target_height=TARGET_HEIGHT,
                target_width=TARGET_WIDTH,
                interpolation=INTERPOLATION,
            )
            processed = replicate_channels(resized, CHANNELS)
            validate_processed_image(processed, filepath, TARGET_SHAPE)

            save_processed_image(processed, output_path)
            records.append(build_processed_record(row, output_path))
        except Exception as exc:
            validation_errors.append(
                {
                    "image_id": int(row["image_id"]),
                    "filepath": filepath,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    processed_df = pd.DataFrame(records)
    processed_df = processed_df.sort_values("image_id").reset_index(drop=True)
    processed_df.to_csv(PROCESSED_INDEX_PATH, index=False)

    end_time = datetime.now(timezone.utc)
    runtime_seconds = perf_counter() - start_counter

    fold_distribution = {}
    if not processed_df.empty:
        for fold, fold_df in processed_df.groupby("fold"):
            fold_distribution[f"fold_{int(fold)}"] = {
                "samples": int(len(fold_df)),
                "patients": int(fold_df["patient_id"].nunique()),
                "class_distribution": {
                    str(k): int(v)
                    for k, v in fold_df["label"].value_counts().sort_index().items()
                },
            }

    stats = {
        "total_images": int(len(registry_df)),
        "processed_images": int(len(processed_df)),
        "failed_images": int(len(validation_errors)),
        "output_shape": list(TARGET_SHAPE),
        "normalization": NORMALIZATION,
        "interpolation": INTERPOLATION,
        "dtype": OUTPUT_DTYPE,
        "channels": CHANNELS,
        "fold_distribution": fold_distribution,
    }

    report = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "runtime_seconds": runtime_seconds,
        "images_processed": int(len(processed_df)),
        "images_failed": int(len(validation_errors)),
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "config": {
            "input_index": INDEX_PATH.as_posix(),
            "split_dir": SPLIT_DIR.as_posix(),
            "output_dir": OUTPUT_DIR.as_posix(),
            "target_height": TARGET_HEIGHT,
            "target_width": TARGET_WIDTH,
            "channels": CHANNELS,
            "normalization": NORMALIZATION,
            "interpolation": INTERPOLATION,
            "dtype": OUTPUT_DTYPE,
            "expected_source_shape": list(EXPECTED_SOURCE_SHAPE),
        },
    }

    write_json(PREPROCESSING_STATS_PATH, stats)
    write_json(PREPROCESSING_REPORT_PATH, report)

    if validation_errors:
        raise RuntimeError(
            f"Preprocessing failed for {len(validation_errors)} images. "
            f"See {PREPROCESSING_REPORT_PATH}."
        )

    print(
        f"[OK] Preprocessed {len(processed_df)} images to "
        f"{TARGET_SHAPE} {OUTPUT_DTYPE}"
    )


if __name__ == "__main__":
    main()
