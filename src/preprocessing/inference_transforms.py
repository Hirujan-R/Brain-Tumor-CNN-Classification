"""
Inference-time preprocessing transforms that match training preprocessing.

CRITICAL: This module ensures inference uses the SAME preprocessing pipeline as training.
"""
import numpy as np
from src.preprocessing.transforms import zscore_normalize, ensure_grayscale, crop_to_brain, replicate_channels, resize_image


def preprocess_for_inference(
    image: np.ndarray,
    target_height: int = 224,
    target_width: int = 224,
    channels: int = 3,
) -> np.ndarray:
    """
    Apply the EXACT same preprocessing pipeline used during training.
    
    Pipeline:
    1. Ensure grayscale
    2. Z-score normalization (NOT min-max!)
    3. Crop to brain region
    4. Resize to target dimensions
    5. Replicate to 3 channels
    
    Args:
        image: Raw loaded image (can be 2D or 3D)
        target_height: Target height (default 224)
        target_width: Target width (default 224)
        channels: Number of output channels (default 3)
    
    Returns:
        Preprocessed image array with shape (H, W, C) ready for model input
    """
    # Step 1: Convert to grayscale if needed
    image = ensure_grayscale(image)
    
    # Step 2: Z-score normalization (CRITICAL - matches training!)
    normalized = zscore_normalize(image)
    
    # Step 3: Crop to brain region
    cropped = crop_to_brain(image=normalized)
    
    # Step 4: Resize to target dimensions
    resized = resize_image(
        image=cropped,
        target_height=target_height,
        target_width=target_width,
        interpolation="bilinear",
        clip_to_0_1=False  # Important: don't clip z-score normalized values!
    )
    
    # Step 5: Replicate to 3 channels
    processed = replicate_channels(resized, channels)
    
    return processed
