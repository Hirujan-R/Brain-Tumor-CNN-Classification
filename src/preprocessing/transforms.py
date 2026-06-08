import numpy as np
from PIL import Image


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    denominator = max_value - min_value
    if denominator <= 0:
        raise ValueError("Cannot min-max normalize a constant image")
    return ((image - min_value) / denominator).astype(np.float32, copy=False)


def resize_image(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    interpolation: str,
) -> np.ndarray:
    if interpolation != "bilinear":
        raise ValueError(f"Unsupported interpolation: {interpolation}")

    pil_image = Image.fromarray(image.astype(np.float32, copy=False))
    resized = pil_image.resize(
        (target_width, target_height),
        resample=Image.Resampling.BILINEAR,
    )
    resized_array = np.asarray(resized, dtype=np.float32)
    return np.clip(resized_array, 0.0, 1.0).astype(np.float32, copy=False)


def replicate_channels(image: np.ndarray, channels: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image before channel replication, got {image.shape}")
    if channels != 3:
        raise ValueError("Only 3-channel replication is supported")
    return np.repeat(image[:, :, None], channels, axis=2).astype(np.float32, copy=False)
