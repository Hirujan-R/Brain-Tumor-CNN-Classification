from typing import List, Dict
import pandas as pd
from collections import defaultdict


def build_registry(filepaths: List[str], parsed_samples: List[Dict]) -> pd.DataFrame:
    """
    Build dataset registry (index table).

    Returns:
        DataFrame with:
        filepath, label, patient_id, has_mask, image_shape
    """

    records = []
    patient_map = defaultdict(list)

    for filepath, sample in zip(filepaths, parsed_samples):

        record = {
            "filepath": filepath,
            "label": int(sample["label"]),
            "patient_id": sample.get("patient_id", None),
            "has_mask": bool(sample.get("has_mask", False)),
            "image_shape": str(sample.get("image_shape"))
        }

        records.append(record)

        # build patient grouping map
        patient_map[record["patient_id"]].append(filepath)

    df = pd.DataFrame(records)

    return df