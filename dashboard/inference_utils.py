"""Inference helpers for the interactive dashboard.

Discovers trained checkpoints, loads the requested architecture, preprocesses
uploaded images consistently with training, runs predictions and Grad-CAM.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.label_mapping import (  # noqa: E402
    MODEL_LABEL_TO_CLASS_NAME,
    MODEL_LABEL_TO_ORIGINAL_LABEL,
)
from src.inference.gradcam import generate_gradcam  # noqa: E402
from src.inference.predict import load_model, predict  # noqa: E402
from src.preprocessing.inference_transforms import preprocess_for_inference  # noqa: E402

CNN_ARCHITECTURES = ["cnn_baseline", "googlenet", "resnet18", "vgg19"]

CHECKPOINT_DIRS = [
    REPO_ROOT / "brain_tumor_results" / "checkpoints",
    REPO_ROOT / "reports" / "checkpoints",
    REPO_ROOT / "checkpoints",
]

RESULT_JSON_DIRS = [
    REPO_ROOT / "brain_tumor_results",
    REPO_ROOT / "reports",
]


def load_cv_metrics() -> dict:
    """Return {architecture: cv_metrics_payload} from the saved JSON results."""
    payloads = {}
    for directory in RESULT_JSON_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*_cv_metrics.json")):
            with open(path) as handle:
                payload = json.load(handle)
            if "model" in payload:
                payloads[payload["model"]] = payload
    return payloads


def cv_accuracy(payloads: dict) -> dict:
    """{architecture: mean CV accuracy} for fine-tuned CNN models."""
    accuracies = {}
    for model, payload in payloads.items():
        if payload.get("approach") != "fine_tune":
            continue
        accuracies[model] = float(payload["aggregate"]["accuracy_mean"])
    return accuracies


def best_fold(payload: dict) -> int:
    """Fold with the lowest validation loss (fallback to fold 0)."""
    per_fold = payload.get("per_fold", [])
    valid = [
        (fold["fold"], fold.get("val_loss"))
        for fold in per_fold
        if fold.get("val_loss") is not None
    ]
    if not valid:
        return 0
    return int(min(valid, key=lambda item: item[1])[0])


def discover_checkpoints() -> dict:
    """{architecture: {fold: path}} for every checkpoint found on disk."""
    found = {}
    for directory in CHECKPOINT_DIRS:
        if not directory.exists():
            continue
        for arch_dir in directory.glob("*_fold*"):
            name = arch_dir.name
            if "_fold" not in name:
                continue
            arch, fold_text = name.rsplit("_fold", 1)
            ckpt = arch_dir / "best_model.pth"
            if not ckpt.exists() or arch not in CNN_ARCHITECTURES:
                continue
            try:
                fold = int(fold_text)
            except ValueError:
                continue
            found.setdefault(arch, {})[fold] = ckpt

    fallback = REPO_ROOT / "model" / "model.pth"
    if fallback.exists() and "googlenet" not in found:
        found.setdefault("googlenet", {})[0] = fallback
    return found


def available_models() -> dict:
    """{architecture: {'path': Path, 'fold': int, 'accuracy': float|None}}."""
    checkpoints = discover_checkpoints()
    payloads = load_cv_metrics()
    accuracies = cv_accuracy(payloads)

    models = {}
    for arch, folds in checkpoints.items():
        payload = payloads.get(arch)
        fold = best_fold(payload) if payload else min(folds)
        if fold not in folds:
            fold = min(folds)
        models[arch] = {
            "path": folds[fold],
            "fold": fold,
            "accuracy": accuracies.get(arch),
        }
    return models


def strongest_model(models: dict) -> str:
    ranked = sorted(
        models.items(),
        key=lambda item: (item[1]["accuracy"] is None, -(item[1]["accuracy"] or 0.0)),
    )
    return ranked[0][0]


def load_uploaded_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """Load an uploaded file as a float32 array.

    `.npy` files are assumed to be already preprocessed (H, W, C) arrays;
    raster images are treated as raw and converted to grayscale (H, W).
    """
    if filename.lower().endswith(".npy"):
        return np.load(io.BytesIO(file_bytes), allow_pickle=False).astype(np.float32)

    from PIL import Image

    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    return np.asarray(image).astype(np.float32)


def prepare_for_model(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a (H, W, 3) float32 model-ready array and a description."""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        return array.astype(np.float32), "preprocessed (H, W, 3)"
    if array.shape == (3, 224, 224):
        return array.transpose(1, 2, 0).astype(np.float32), "preprocessed (C, H, W)"
    processed = preprocess_for_inference(array)
    return processed, "raw image -> training preprocessing"


def display_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image to [0, 1] RGB for display."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array[:, :, 0]
    minimum, maximum = float(array.min()), float(array.max())
    normalized = (array - minimum) / (maximum - minimum) if maximum > minimum else array * 0
    return np.stack([normalized] * 3, axis=-1)


def predict_with_model(
    model,
    arch: str,
    image: np.ndarray,
    device: str = "cpu",
    checkpoint: str = "",
) -> dict:
    """Run a loaded model on a (H, W, 3) image and return a result dict."""
    result = predict(model, image, device=device)

    predicted = int(result["predicted_label"])
    probabilities = np.asarray(result["probabilities"], dtype=float)

    classes = [
        MODEL_LABEL_TO_CLASS_NAME[index] for index in sorted(MODEL_LABEL_TO_CLASS_NAME)
    ]
    ranking = sorted(
        (
            {
                "class": classes[index],
                "label": index,
                "probability": float(probabilities[index]),
            }
            for index in range(len(classes))
        ),
        key=lambda row: row["probability"],
    )

    return {
        "arch": arch,
        "checkpoint": checkpoint,
        "predicted_label": predicted,
        "predicted_class": MODEL_LABEL_TO_CLASS_NAME[predicted],
        "original_label": int(MODEL_LABEL_TO_ORIGINAL_LABEL[predicted]),
        "probabilities": probabilities,
        "classes": classes,
        "ranking": ranking,
        "model": model,
    }


def predict_image(
    models: dict,
    arch: str,
    image: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Load the requested checkpoint and run a single prediction."""
    path = models[arch]["path"]
    model = load_model(str(path), arch=arch, device=device)
    return predict_with_model(model, arch, image, device=device, checkpoint=str(path))


def gradcam_overlay(model, arch: str, image: np.ndarray, device: str = "cpu"):
    """Return an RGB Grad-CAM overlay for a (H, W, 3) image (or None)."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    cam, _, _ = generate_gradcam(model, image, arch=arch, device=device)
    base = display_image(image)
    overlay = show_cam_on_image(base, cam, use_rgb=True)
    return overlay
