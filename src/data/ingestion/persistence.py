from pathlib import Path
import json
import pandas as pd
from typing import Dict


def persist_outputs(
    outputs: Dict,
    config
) -> None:
    """
    Persist registry outputs to disk.

    Writes:
        - index.csv
        - patient_map.json
        - label_stats.json
        - summary.json
    """

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1. Save registry CSV
    # ----------------------------
    df: pd.DataFrame = outputs["registry_df"]
    csv_path = output_dir / config.index_file
    df.to_csv(csv_path, index=False)

    # ----------------------------
    # 2. Save patient map
    # ----------------------------
    patient_map_path = output_dir / "patient_map.json"
    with open(patient_map_path, "w") as f:
        json.dump(outputs["patient_map"], f, indent=2)

    # ----------------------------
    # 3. Save label stats
    # ----------------------------
    label_stats_path = output_dir / "label_stats.json"
    with open(label_stats_path, "w") as f:
        json.dump(outputs["label_stats"], f, indent=2)

    # ----------------------------
    # 4. Save dataset summary
    # ----------------------------
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(outputs["summary"], f, indent=2)