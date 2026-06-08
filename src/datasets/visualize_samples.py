import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .brain_tumor_dataset import BrainTumorDataset
    from .dataset_utils import filter_registry_by_fold
    from .label_mapping import decode_label
except ImportError:
    from brain_tumor_dataset import BrainTumorDataset
    from dataset_utils import filter_registry_by_fold
    from label_mapping import decode_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a grid of processed MRI samples.")
    parser.add_argument("--index", default="data/processed/processed_index.csv")
    parser.add_argument("--output", default="reports/sample_grid.png")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.index)
    if args.fold is not None:
        registry = filter_registry_by_fold(registry, fold=args.fold, split="val")

    dataset = BrainTumorDataset(
        processed_index_path=args.index,
        registry_df=registry,
        return_metadata=True,
    )

    rng = random.Random(args.seed)
    sample_count = min(args.num_samples, len(dataset))
    indices = rng.sample(range(len(dataset)), sample_count)

    columns = 4
    rows = (sample_count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes = axes.flatten() if sample_count > 1 else [axes]

    for axis in axes:
        axis.axis("off")

    for axis, index in zip(axes, indices):
        image, label, metadata = dataset[index]
        image_hwc = image.permute(1, 2, 0).numpy()
        axis.imshow(image_hwc[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"{decode_label(int(label))}\n"
            f"patient={metadata['patient_id']} fold={metadata['fold']}",
            fontsize=8,
        )
        axis.axis("off")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved sample grid to {output_path}")


if __name__ == "__main__":
    main()
