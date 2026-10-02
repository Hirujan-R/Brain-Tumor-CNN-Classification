#!/usr/bin/env python
"""Quantify Grad-CAM localization against the ground-truth tumor masks.

For a sample of images, each fine-tuned CNN's Grad-CAM (for the true class) is
compared to the tumor mask that was carved out of the raw Figshare ``.mat`` file
and warped through the exact training preprocessing (crop + resize).

Metrics per model:
  * pointing accuracy  - is the CAM's peak pixel inside the tumor?
  * energy in mask     - fraction of CAM mass inside the tumor
  * ROC-AUC            - CAM as a detector of tumor pixels
  * IoU@top20 / Dice@top20 - overlap using the top 20% CAM pixels
  * IoU@0.5 / Dice@0.5 - overlap using CAM >= 0.5

Usage:
    python scripts/evaluate_gradcam.py --num-samples 40 --device cpu
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.inference_utils import available_models, load_model  # noqa: E402
from src.datasets.label_mapping import encode_label  # noqa: E402
from src.inference.gradcam import generate_gradcam  # noqa: E402
from src.preprocessing.transforms import ensure_grayscale  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Grad-CAM vs tumor masks.")
    parser.add_argument("--results-dir", default="brain_tumor_results")
    parser.add_argument("--index", default="data/processed/processed_index.csv")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", default="auto", help="CAM method (auto, gradcam, gradcam++, xgradcam, hirescam, layercam)")
    parser.add_argument("--output-dir", default="brain_tumor_results/gradcam_eval")
    parser.add_argument("--crop-threshold", type=float, default=0.05)
    return parser.parse_args()


def load_mat_and_mask(filepath: str):
    with h5py.File(filepath, "r") as handle:
        cjdata = handle["cjdata"]
        image = np.array(cjdata["image"]).T.astype(np.float32)
        mask = None
        if "tumorMask" in cjdata:
            mask = np.array(cjdata["tumorMask"]).T.astype(np.float32)
    return image, mask


def aligned_mask(image: np.ndarray, mask: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Warp a raw tumor mask through the same crop+resize used for training."""
    gray = ensure_grayscale(image)
    foreground = gray > threshold
    if not foreground.any():
        return np.zeros((224, 224), dtype=bool)

    rows = np.any(foreground, axis=1)
    cols = np.any(foreground, axis=0)
    y = np.where(rows)[0]
    x = np.where(cols)[0]
    if len(y) == 0 or len(x) == 0:
        return np.zeros((224, 224), dtype=bool)

    pad = 10
    ymin, ymax = max(0, y[0] - pad), min(gray.shape[0], y[-1] + pad)
    xmin, xmax = max(0, x[0] - pad), min(gray.shape[1], x[-1] + pad)

    cropped = mask[ymin:ymax, xmin:xmax]
    resized = np.asarray(
        Image.fromarray(cropped.astype(np.float32)).resize(
            (224, 224), resample=Image.Resampling.BILINEAR
        )
    )
    return resized >= 0.5


def cam_metrics(cam: np.ndarray, mask: np.ndarray) -> dict:
    cam = np.asarray(cam, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return None

    peak = np.unravel_index(np.argmax(cam), cam.shape)
    pointing = float(mask[peak])

    total = cam.sum()
    energy = float(cam[mask].sum() / total) if total > 0 else 0.0

    auc = float(roc_auc_score(mask.ravel(), cam.ravel()))

    flat = cam.ravel()
    k = max(1, int(0.2 * flat.size))
    top20_threshold = np.partition(flat, -k)[-k]
    binary_top = cam >= top20_threshold
    binary_half = cam >= 0.5

    inter_top = np.logical_and(binary_top, mask).sum()
    union_top = np.logical_or(binary_top, mask).sum()
    iou_top = float(inter_top / union_top) if union_top else 1.0
    dice_top = float(2 * inter_top / (binary_top.sum() + mask.sum()))

    inter_half = np.logical_and(binary_half, mask).sum()
    union_half = np.logical_or(binary_half, mask).sum()
    iou_half = float(inter_half / union_half) if union_half else 1.0
    dice_half = float(2 * inter_half / (binary_half.sum() + mask.sum()))

    return {
        "pointing_accuracy": pointing,
        "energy_in_mask": energy,
        "roc_auc": auc,
        "iou_top20": iou_top,
        "dice_top20": dice_top,
        "iou_0.5": iou_half,
        "dice_0.5": dice_half,
    }


def figure_results(summary: pd.DataFrame, output_dir: Path, model_names) -> None:
    metrics = ["pointing_accuracy", "energy_in_mask", "roc_auc", "iou_top20", "dice_top20"]
    labels = {
        "pointing_accuracy": "Pointing accuracy",
        "energy_in_mask": "CAM energy in mask",
        "roc_auc": "ROC-AUC",
        "iou_top20": "IoU @ top 20%",
        "dice_top20": "Dice @ top 20%",
    }
    table = summary.set_index("model").loc[model_names]
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    for index, model in enumerate(model_names):
        values = [table.loc[model, metric] for metric in metrics]
        ax.bar(x + index * width - 0.4 + width / 2, values, width, label=model, color=colors[index])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in metrics], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Grad-CAM localization vs ground-truth tumor mask")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "gradcam_localization_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def figure_examples(examples: list, output_dir: Path, model_names) -> None:
    if not examples:
        return
    rows = len(examples)
    cols = len(model_names) + 2
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for row, example in enumerate(examples):
        axes[row, 0].imshow(example["image"], cmap="gray")
        axes[row, 0].set_title("Image")
        axes[row, 1].imshow(example["image"], cmap="gray")
        axes[row, 1].imshow(np.ma.masked_where(~example["mask"], example["mask"]), cmap="autumn", alpha=0.5)
        axes[row, 1].set_title("Tumor mask")
        for col, model in enumerate(model_names, start=2):
            axes[row, col].imshow(example["image"], cmap="gray")
            axes[row, col].imshow(example["cams"][model], cmap="jet", alpha=0.45)
            axes[row, col].set_title(model, fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Grad-CAM vs tumor mask (same sample, all models)", y=1.0)
    fig.tight_layout()
    path = output_dir / "gradcam_examples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = available_models()
    if not models:
        raise SystemExit("No checkpoints found; train models first.")
    model_names = list(models.keys())
    print("Models:", model_names)

    # Load models once
    loaded = {
        name: load_model(str(info["path"]), arch=name, device=args.device)
        for name, info in models.items()
    }

    index = pd.read_csv(args.index)
    rng = np.random.default_rng(args.seed)
    candidate_rows = index.sample(
        n=min(len(index), args.num_samples * 4), random_state=args.seed
    )

    records = []
    examples = []
    evaluated = 0
    for _, row in candidate_rows.iterrows():
        if evaluated >= args.num_samples:
            break
        filepath = row.get("original_path")
        if not isinstance(filepath, str) or not Path(filepath).exists():
            continue
        raw_image, raw_mask = load_mat_and_mask(filepath)
        if raw_mask is None or raw_mask.sum() == 0:
            continue
        mask = aligned_mask(raw_image, raw_mask, threshold=args.crop_threshold)
        if mask.sum() == 0:
            continue

        processed = np.load(row["processed_path"]).astype(np.float32)
        if processed.ndim == 2:
            processed = np.repeat(processed[:, :, None], 3, axis=2)
        image_display = processed[:, :, 0]
        image_display = (image_display - image_display.min()) / (
            np.ptp(image_display) + 1e-8
        )

        true_label = encode_label(int(row["label"]))
        example = {"image": image_display, "mask": mask, "cams": {}}

        for name in model_names:
            model = loaded[name]
            try:
                cam, _, _ = generate_gradcam(
                    model,
                    processed,
                    target_class=true_label,
                    device=args.device,
                    arch=name,
                    method=args.method,
                )
                metrics = cam_metrics(cam, mask)
                if metrics is None:
                    continue
                metrics.update({"model": name, "image_id": int(row["image_id"])})
                records.append(metrics)
                example["cams"][name] = cam
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] {name} failed on image {row['image_id']}: {exc}")

        if len(example["cams"]) == len(model_names):
            examples.append(example)
        evaluated += 1
        if evaluated % 10 == 0:
            print(f"evaluated {evaluated} samples")

    if not records:
        raise SystemExit("No samples evaluated.")

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "gradcam_per_sample.csv", index=False)

    metric_cols = [
        "pointing_accuracy",
        "energy_in_mask",
        "roc_auc",
        "iou_top20",
        "dice_top20",
        "iou_0.5",
        "dice_0.5",
    ]
    summary = (
        frame.groupby("model")[metric_cols]
        .mean()
        .reset_index()
        .sort_values("energy_in_mask", ascending=False)
    )
    summary.insert(1, "n_samples", frame.groupby("model").size().reindex(summary["model"]).values)
    summary.to_csv(output_dir / "gradcam_summary.csv", index=False)

    print("\nGrad-CAM localization vs tumor mask (means)")
    print(summary.to_string(index=False))

    figure_results(summary, output_dir, model_names)
    figure_examples(examples[:3], output_dir, model_names)


if __name__ == "__main__":
    main()
