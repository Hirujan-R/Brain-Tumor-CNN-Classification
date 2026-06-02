import pandas as pd
from typing import Dict
import json
from collections import Counter


def create_registry_outputs(df: pd.DataFrame) -> Dict:
    """
    Convert registry DataFrame into structured outputs
    for persistence layer.

    Returns:
        dict containing:
            - cleaned_df
            - patient_map
            - label_stats
    """

    # ----------------------------
    # 1. Clean dataframe (safety step)
    # ----------------------------
    df = df.copy()

    # Ensure correct types
    df["label"] = df["label"].astype(int)

    # ----------------------------
    # 2. Build patient map
    # ----------------------------
    patient_map = (
        df.groupby("patient_id")["filepath"]
        .apply(list)
        .to_dict()
    )

    # ----------------------------
    # 3. Label distribution stats
    # ----------------------------
    label_stats = Counter(df["label"])

    label_stats = {
        str(k): int(v) for k, v in label_stats.items()
    }

    # ----------------------------
    # 4. Dataset summary metadata
    # ----------------------------
    summary = {
        "num_samples": len(df),
        "num_patients": len(patient_map),
        "num_classes": len(label_stats)
    }

    return {
        "registry_df": df,
        "patient_map": patient_map,
        "label_stats": label_stats,
        "summary": summary
    }