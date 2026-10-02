"""
Inference-time preprocessing transforms that match training preprocessing.

CRITICAL: This module ensures inference uses the SAME preprocessing pipeline as
training (see ``src/preprocessing/preprocess.py``).

Training order:
    1. grayscale
    2. crop to brain (on raw values)
    3. resize
    4. z-score normalize
    5. replicate to 3 channels
"""

import numpy as np

from src.preprocessing.transforms import (
    crop_to_brain,
    ensure_grayscale,
    replicate_channels,
    resize_image,
    zscore_normalize,
)


def preprocess_for_inference(
    image: np.ndarray,
    target_height: int = 224,
    target_width: int = 224,
    channels: int = 3,
) -> np.ndarray:
    """
    Apply the EXACT same preprocessing pipeline used during training.

    Returns:
        Preprocessed image array with shape (H, W, C) ready for model input.
    """
    # Step 1: Convert to grayscale if needed
    image = ensure_grayscale(image)

    # Step 2: Crop to brain region (on raw values)
    cropped = crop_to_brain(image=image)

    # Step 3: Resize to target dimensions (do NOT clip - z-score follows)
    resized = resize_image(
        image=cropped,
        target_height=target_height,
        target_width=target_width,
        interpolation="bilinear",
        clip_to_0_1=False,
    )

    # Step 4: Z-score normalization (matches training)
    normalized = zscore_normalize(resized)

    # Step 5: Replicate to 3 channels
    return replicate_channels(normalized, channels)
