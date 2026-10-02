"""Inference utilities.

The public API is:

    load_model(model_path, arch="googlenet", ...)
    predict(model, image, ..., preprocess_raw=False) -> dict

`predict` accepts either an already-preprocessed array shaped (H, W, C) / (C, H, W),
or a raw 2D image when ``preprocess_raw=True``. It returns a dictionary with the
predicted model label, original Figshare label, class name and probabilities.
"""

import numpy as np
import torch

from src.datasets.label_mapping import (
    MODEL_LABEL_TO_CLASS_NAME,
    MODEL_LABEL_TO_ORIGINAL_LABEL,
)
from src.models.factory import build_classifier
from src.preprocessing.inference_transforms import preprocess_for_inference


def load_model(
    model_path: str = "model/model.pth",
    arch: str = "googlenet",
    num_classes: int = 3,
    device="cpu",
):
    """Build the requested architecture and load a saved state dict."""
    model = build_classifier(
        arch, num_classes=num_classes, pretrained=False, aux_logits=True
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_raw_image(
    image: np.ndarray,
    target_height: int = 224,
    target_width: int = 224,
    channels: int = 3,
    clip_compatibility: bool = False,
) -> np.ndarray:
    """
    Apply the exact preprocessing used during training to a raw image.

    With ``clip_compatibility=True`` the z-score output is clipped to [0, 1]
    (legacy behaviour); by default the full z-score range is preserved.
    """
    processed = preprocess_for_inference(
        np.asarray(image),
        target_height=target_height,
        target_width=target_width,
        channels=channels,
    )
    if clip_compatibility:
        processed = np.clip(processed, 0.0, 1.0)
    return processed


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)

    if array.ndim == 2:
        raise ValueError(
            f"Expected a 3D image (H, W, C) or (C, H, W); got shape {array.shape}. "
            "Pass preprocess_raw=True for raw 2D images."
        )
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {array.shape}")

    if array.shape[2] in (1, 3, 4):
        # HWC -> CHW
        array = array.transpose(2, 0, 1)
    elif array.shape[0] not in (1, 3, 4):
        raise ValueError(f"Cannot infer channel axis for shape {array.shape}")

    return torch.from_numpy(np.ascontiguousarray(array))


def predict(
    model,
    image_tensor,
    device="cpu",
    preprocess_raw: bool = False,
) -> dict:
    """Run a single-image prediction and return a result dictionary."""
    if not isinstance(image_tensor, (np.ndarray, torch.Tensor)):
        raise TypeError("image_tensor must be a numpy array or torch.Tensor.")

    if preprocess_raw:
        image_tensor = preprocess_raw_image(np.asarray(image_tensor))

    if isinstance(image_tensor, torch.Tensor):
        tensor = image_tensor.to(dtype=torch.float32)
    else:
        tensor = _to_tensor(image_tensor)

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if hasattr(outputs, "logits"):
            outputs = outputs.logits
        probs = torch.softmax(outputs, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        probs_np = probs.squeeze(0).cpu().numpy()

    return {
        "predicted_label": pred_class,
        "original_label": int(MODEL_LABEL_TO_ORIGINAL_LABEL[pred_class]),
        "class_name": MODEL_LABEL_TO_CLASS_NAME[pred_class],
        "probabilities": probs_np,
    }
