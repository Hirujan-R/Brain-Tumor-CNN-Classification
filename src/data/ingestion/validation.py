import numpy as np
from typing import Dict, Tuple


def validate_sample(sample: Dict) -> Tuple[bool, str]:
    """
    Validate parsed MRI sample.

    Returns:
        (is_valid, reason_if_invalid)
    """

    # ----------------------------
    # 1. Check image exists
    # ----------------------------
    image = sample.get("image", None)
    if image is None:
        return False, "Missing image"

    if not isinstance(image, np.ndarray):
        return False, "Image is not numpy array"

    if image.size == 0:
        return False, "Empty image"

    # ----------------------------
    # 2. Check label
    # ----------------------------
    label = sample.get("label", None)
    if label is None:
        return False, "Missing label"

    if not isinstance(label, (int, np.integer)):
        return False, "Label is not integer"

    if label not in [1, 2, 3]:
        return False, f"Invalid label: {label}"

    # ----------------------------
    # 3. Check image quality
    # ----------------------------
    if np.isnan(image).any():
        return False, "Image contains NaNs"

    if np.std(image) == 0:
        return False, "Image has zero variance"

    # ----------------------------
    # 4. Sanity shape check
    # ----------------------------
    if len(image.shape) != 2:
        return False, f"Invalid image shape: {image.shape}"

    if image.shape[0] < 64 or image.shape[1] < 64:
        return False, f"Image too small: {image.shape}"

    return True, "valid"