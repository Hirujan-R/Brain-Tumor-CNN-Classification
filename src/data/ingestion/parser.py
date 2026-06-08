import h5py
import numpy as np
from typing import Dict, Any


def parse_mat_file(filepath: str) -> Dict[str, Any]:
    """
    Parser for MATLAB v7.3 (.mat HDF5 format)
    Compatible with Brain Tumor dataset.
    """

    with h5py.File(filepath, "r") as f:

        # ----------------------------
        # Access cjdata group
        # ----------------------------
        cjdata = f["cjdata"]

        # ----------------------------
        # IMAGE
        # ----------------------------
        image = np.array(cjdata["image"]).T  # transpose is important

        # ----------------------------
        # LABEL
        # ----------------------------
        label = int(np.array(cjdata["label"]).squeeze())

        # ----------------------------
        # PID (IMPORTANT FOR YOUR PIPELINE)
        # ----------------------------
        pid = np.array(cjdata["PID"]).squeeze()

        # ----------------------------
        # MASK (optional)
        # ----------------------------
        mask = None
        has_mask = False

        if "tumorMask" in cjdata:
            mask = np.array(cjdata["tumorMask"]).T
            has_mask = True

        # ----------------------------
        # OUTPUT STANDARDIZATION
        # ----------------------------
        return {
            "image": image,
            "label": label,
            "patient_id": pid,
            "mask": mask,
            "has_mask": has_mask,
            "image_shape": image.shape
        }

