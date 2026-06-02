from pathlib import Path
import re
from typing import Optional


def extract_patient_id(filepath: str) -> str:
    """
    Extract patient ID from file path.

    Strategy:
    - Try filename pattern first
    - Fallback to parent folder
    - Final fallback: stable hash-like fallback
    """

    path = Path(filepath)

    filename = path.stem  # without .mat

    # ----------------------------
    # Strategy 1: numeric ID in filename
    # e.g., 00123.mat → 00123
    # ----------------------------
    match = re.search(r"\d+", filename)
    if match:
        return match.group(0)

    # ----------------------------
    # Strategy 2: parent folder name
    # ----------------------------
    parent = path.parent.name
    if parent and parent != "extracted_mat":
        return parent

    # ----------------------------
    # Strategy 3: fallback (last resort)
    # ----------------------------
    return filename