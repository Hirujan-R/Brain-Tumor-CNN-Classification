import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    denominator = max_value - min_value
    if denominator <= 0:
        raise ValueError("Cannot min-max normalize a constant image")
    return ((image - min_value) / denominator).astype(np.float32, copy=False)

def zscore_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std < 1e-8:
        return np.zeros_like(image)

    return (image - mean) / std


def resize_image(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    interpolation: str,
    clip_to_0_1: bool = True,
) -> np.ndarray:
    if interpolation != "bilinear":
        raise ValueError(f"Unsupported interpolation: {interpolation}")

    pil_image = Image.fromarray(image.astype(np.float32, copy=False))
    resized = pil_image.resize(
        (target_width, target_height),
        resample=Image.Resampling.BILINEAR,
    )
    resized_array = np.asarray(resized, dtype=np.float32)
    if clip_to_0_1:
        resized_array = np.clip(resized_array, 0.0, 1.0)
    return resized_array.astype(np.float32, copy=False)


def replicate_channels(image: np.ndarray, channels: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image before channel replication, got {image.shape}")
    if channels != 3:
        raise ValueError("Only 3-channel replication is supported")
    return np.repeat(image[:, :, None], channels, axis=2).astype(np.float32, copy=False)

import numpy as np


def crop_background(image: np.ndarray, threshold: float = 0.05):
    image = np.asarray(image)

    mask = image > threshold

    if not np.any(mask):
        return image  # IMPORTANT fallback

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return image

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    if len(y_indices) == 0 or len(x_indices) == 0:
        return image

    ymin, ymax = y_indices[0], y_indices[-1]
    xmin, xmax = x_indices[0], x_indices[-1]

    # safety padding
    pad = 5
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    ymax = min(image.shape[0] - 1, ymax + pad)
    xmax = min(image.shape[1] - 1, xmax + pad)

    cropped = image[ymin:ymax+1, xmin:xmax+1]

    # final safety check
    if cropped.size == 0:
        return image

    return cropped.astype(np.float32, copy=False)


def extract_brain(
    image: np.ndarray,
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        brain_only
        brain_mask
    """

    binary = image > threshold

    labels, num_labels = ndimage.label(binary)

    if num_labels == 0:
        return image, np.zeros_like(image)

    component_sizes = ndimage.sum(
        binary,
        labels,
        range(1, num_labels + 1)
    )

    largest_component = np.argmax(component_sizes) + 1

    brain_mask = labels == largest_component

    brain_only = image * brain_mask

    return brain_only, brain_mask.astype(np.uint8)


def clean_brain_mask(
    mask: np.ndarray,
    iterations: int = 3,
):
    mask = ndimage.binary_closing(
        mask,
        iterations=iterations
    )

    mask = ndimage.binary_fill_holes(mask)

    return mask

def skull_strip(
    image: np.ndarray,
    threshold: float = 0.1,
):
    brain_only, mask = extract_brain(
        image,
        threshold
    )

    mask = clean_brain_mask(mask)

    brain_only = image * mask

    return brain_only, mask

def skull_strip_and_crop(
    image: np.ndarray,
    threshold: float = 0.1,
):
    brain_only, mask = skull_strip(
        image,
        threshold
    )

    cropped = crop_background(
        brain_only,
        threshold=threshold
    )

    return cropped

def crop_to_brain(image: np.ndarray, threshold: float = 0.05):
    mask = image > threshold

    if not np.any(mask):
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y = np.where(rows)[0]
    x = np.where(cols)[0]

    if len(y) == 0 or len(x) == 0:
        return image

    pad = 10

    ymin, ymax = max(0, y[0]-pad), min(image.shape[0], y[-1]+pad)
    xmin, xmax = max(0, x[0]-pad), min(image.shape[1], x[-1]+pad)


    return image[ymin:ymax, xmin:xmax]

def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim == 3:
        # if RGB-like (H,W,3), convert to grayscale
        image = image.mean(axis=2)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image after grayscale conversion, got {image.shape}")

    return image.astype(np.float32, copy=False)

def safe_imshow(ax, img, title):
    if img is None:
        ax.set_title(f"{title} (None)")
        return

    img = np.asarray(img)

    if img.size == 0:
        ax.set_title(f"{title} (empty)")
        return

    img = np.nan_to_num(img)

    if img.ndim != 2:
        img = img.squeeze()

    ax.imshow(img, cmap="gray")
    ax.set_title(title)


def show_preprocessing(image):
    image = ensure_grayscale(image)

    stripped, mask = skull_strip(image)
    cropped = crop_background(stripped)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    safe_imshow(ax[0], image, "Original")
    safe_imshow(ax[1], mask, "Brain Mask")
    safe_imshow(ax[2], cropped, "Cropped")

    plt.show()


def main():
    image = np.load("data/processed/images/000896.npy")
    show_preprocessing(image)

if __name__ == "__main__":
    main()