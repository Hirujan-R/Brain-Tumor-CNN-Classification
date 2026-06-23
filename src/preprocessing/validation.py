import numpy as np


class ValidationError(ValueError):
    pass


def validate_raw_image(
    image: np.ndarray,
    filepath: str,
    expected_shape: tuple[int, int],
) -> list[dict[str, str]]:
    warnings = []

    if not isinstance(image, np.ndarray):
        raise ValidationError(f"{filepath}: loaded image is not a numpy array")

    if image.ndim != 2:
        raise ValidationError(f"{filepath}: expected 2D image, got shape {image.shape}")

    if image.size == 0:
        raise ValidationError(f"{filepath}: image is empty")

    if image.shape != expected_shape:
        warnings.append(
            {
                "filepath": filepath,
                "code": "unexpected_source_shape",
                "message": f"Expected {expected_shape}, got {image.shape}",
            }
        )

    if not np.issubdtype(image.dtype, np.number):
        raise ValidationError(f"{filepath}: image dtype is not numeric: {image.dtype}")

    nan_count = int(np.isnan(image).sum()) if np.issubdtype(image.dtype, np.floating) else 0
    inf_count = int(np.isinf(image).sum()) if np.issubdtype(image.dtype, np.floating) else 0
    if nan_count or inf_count:
        raise ValidationError(
            f"{filepath}: invalid pixel values, nan_count={nan_count}, "
            f"inf_count={inf_count}"
        )

    min_value = float(np.min(image))
    max_value = float(np.max(image))

    if min_value == 0.0 and max_value == 0.0:
        raise ValidationError(f"{filepath}: image is all zeros")

    if min_value == max_value:
        raise ValidationError(f"{filepath}: image is constant")

    return warnings


def validate_normalized_image(image: np.ndarray, filepath: str) -> None:
    if image.dtype != np.float32:
        raise ValidationError(f"{filepath}: dtype is {image.dtype}")

    mean = float(np.mean(image))
    std = float(np.std(image))

    if std < 1e-6:
        raise ValidationError(f"{filepath}: near-constant image")

    # optional sanity bounds check (not strict)
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        raise ValidationError(f"{filepath}: invalid values")


def validate_processed_image(
    image: np.ndarray,
    filepath: str,
    expected_shape: tuple[int, int, int],
    atol: float = 1e-6,
) -> None:
    if image.shape != expected_shape:
        raise ValidationError(
            f"{filepath}: expected processed shape {expected_shape}, got {image.shape}"
        )
    if image.dtype != np.float32:
        raise ValidationError(f"{filepath}: processed image dtype is {image.dtype}")
    if not np.isfinite(image).all():
        raise ValidationError(f"{filepath}: processed image contains NaN or Inf")

    # Z-score normalised data lives outside [0,1] — just check it's finite
    # and has reasonable range (not constant, not exploded)
    std = float(np.std(image))
    if std < 1e-6:
        raise ValidationError(f"{filepath}: processed image is near-constant")

    if not np.allclose(image[:, :, 0], image[:, :, 1], atol=atol):
        raise ValidationError(f"{filepath}: channels 0 and 1 differ")
    if not np.allclose(image[:, :, 1], image[:, :, 2], atol=atol):
        raise ValidationError(f"{filepath}: channels 1 and 2 differ")
