import os
from typing import List, Dict, Tuple

import pandas as pd

# Import your modular layers
from config import IngestionConfig
from loader import discover_mat_files
from parser import parse_mat_file
from validation import validate_sample
from extract_patient_id import extract_patient_id
from create_registry_outputs import create_registry_outputs
from persistence import persist_outputs


# ----------------------------
# MAIN INGESTION PIPELINE
# ----------------------------

def run_ingestion(config: IngestionConfig) -> None:
    """
    Full ingestion pipeline orchestration.

    Steps:
        1. Discover .mat files
        2. Parse each file
        3. Validate samples
        4. Extract patient IDs
        5. Build registry DataFrame
        6. Create outputs (stats, maps)
        7. Persist to disk
    """

    print("\n[INFO] Starting ingestion pipeline...\n")

    # ----------------------------
    # 1. FILE DISCOVERY
    # ----------------------------
    print("[1/6] Discovering .mat files...")
    filepaths = discover_mat_files(config.input_dir)

    print(f"[INFO] Found {len(filepaths)} files")

    # ----------------------------
    # 2. PARSE + VALIDATE LOOP
    # ----------------------------
    print("\n[2/6] Parsing and validating files...")

    valid_samples = []
    valid_filepaths = []

    skipped = 0

    for fp in filepaths:

        try:
            sample = parse_mat_file(fp)

            is_valid, reason = validate_sample(sample)

            if not is_valid:
                skipped += 1
                continue

            # ----------------------------
            # 3. PATIENT ID EXTRACTION
            # ----------------------------
            patient_id = extract_patient_id(fp)
            sample["patient_id"] = patient_id

            valid_samples.append(sample)
            valid_filepaths.append(fp)

        except Exception as e:
            skipped += 1
            continue

    print(f"[INFO] Valid samples: {len(valid_samples)}")
    print(f"[INFO] Skipped samples: {skipped}")

    if len(valid_samples) == 0:
        raise RuntimeError("No valid samples found. Ingestion failed.")

    # ----------------------------
    # 4. BUILD REGISTRY DATAFRAME
    # ----------------------------
    print("\n[3/6] Building registry...")

    records = []

    for fp, sample in zip(valid_filepaths, valid_samples):

        records.append({
            "filepath": fp,
            "label": int(sample["label"]),
            "patient_id": sample["patient_id"],
            "has_mask": bool(sample["has_mask"]),
            "image_shape": sample["image_shape"]
        })

    df = pd.DataFrame(records)

    print(f"[INFO] Registry built with {len(df)} samples")

    # ----------------------------
    # 5. CREATE REGISTRY OUTPUTS
    # ----------------------------
    print("\n[4/6] Creating registry outputs...")

    outputs = create_registry_outputs(df)

    print("[INFO] Registry outputs created")

    # ----------------------------
    # 6. PERSIST OUTPUTS
    # ----------------------------
    print("\n[5/6] Persisting outputs...")

    persist_outputs(outputs, config)

    print(f"[INFO] Outputs saved to: {config.output_dir}")

    # ----------------------------
    # FINAL SUMMARY
    # ----------------------------
    print("\n[6/6] Ingestion complete!\n")
    print("========== SUMMARY ==========")
    print(f"Total files discovered: {len(filepaths)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Skipped files: {skipped}")
    print(f"Patients: {len(outputs['patient_map'])}")
    print(f"Classes: {outputs['summary']['num_classes']}")
    print("=============================\n")


# ----------------------------
# ENTRY POINT
# ----------------------------

if __name__ == "__main__":

    config = IngestionConfig()

    run_ingestion(config)