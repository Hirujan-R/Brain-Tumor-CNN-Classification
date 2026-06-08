from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .label_mapping import encode_label
except ImportError:
    from label_mapping import encode_label


REQUIRED_COLUMNS = {
    "image_id",
    "patient_id",
    "label",
    "processed_path",
    "fold",
}

EXPECTED_IMAGE_SHAPE = (224, 224, 3)
EXPECTED_TENSOR_SHAPE = (3, 224, 224)


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for preprocessed Figshare brain tumor MRI arrays.

    The dataset reads model-ready .npy images produced by Stage 4 and converts
    original Figshare labels into deterministic zero-based model labels.
    """

    def __init__(
        self,
        processed_index_path: str | Path = "data/processed/processed_index.csv",
        registry_df: pd.DataFrame | None = None,
        transform: Any | None = None,
        target_transform: Any | None = None,
        return_metadata: bool = True,
        validate_files: bool = True,
        validate_samples: bool = True,
    ) -> None:
        if registry_df is None:
            self.registry = pd.read_csv(processed_index_path)
        else:
            self.registry = registry_df.copy()

        self.processed_index_path = Path(processed_index_path)
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata
        self.validate_samples = validate_samples

        self._validate_registry_columns()
        self.registry = self.registry.reset_index(drop=True)

        if validate_files:
            self._validate_files_exist()

    def __len__(self) -> int:
        return len(self.registry)

    def __getitem__(self, index: int):
        row = self.registry.iloc[index]
        image = self._load_image(row["processed_path"])
        label = encode_label(row["label"])

        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        tensor = tensor.to(dtype=torch.float32)

        if self.validate_samples:
            self._validate_tensor(tensor, label, row["processed_path"])

        if self.transform is not None:
            tensor = self.transform(tensor)

        if self.target_transform is not None:
            label = self.target_transform(label)

        label_tensor = torch.tensor(label, dtype=torch.long)

        if not self.return_metadata:
            return tensor, label_tensor

        metadata = {
            "patient_id": str(row["patient_id"]),
            "image_id": int(row["image_id"]),
            "fold": int(row["fold"]),
            "processed_path": str(row["processed_path"]),
        }
        if "original_path" in row:
            metadata["original_path"] = str(row["original_path"])

        return tensor, label_tensor, metadata

    def _validate_registry_columns(self) -> None:
        missing = REQUIRED_COLUMNS - set(self.registry.columns)
        if missing:
            raise ValueError(f"Processed registry missing columns: {sorted(missing)}")

    def _validate_files_exist(self) -> None:
        missing_paths = [
            path
            for path in self.registry["processed_path"]
            if not Path(path).exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Missing processed image files: {missing_paths[:10]}"
            )

    def _load_image(self, processed_path: str) -> np.ndarray:
        image = np.load(processed_path, allow_pickle=False)
        if image.shape != EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"{processed_path}: expected shape {EXPECTED_IMAGE_SHAPE}, "
                f"got {image.shape}"
            )
        if image.dtype != np.float32:
            raise TypeError(f"{processed_path}: expected float32, got {image.dtype}")
        return image

    def _validate_tensor(
        self,
        tensor: torch.Tensor,
        label: int,
        processed_path: str,
    ) -> None:
        if tuple(tensor.shape) != EXPECTED_TENSOR_SHAPE:
            raise ValueError(
                f"{processed_path}: expected tensor shape {EXPECTED_TENSOR_SHAPE}, "
                f"got {tuple(tensor.shape)}"
            )

        if tensor.dtype != torch.float32:
            raise TypeError(f"{processed_path}: expected torch.float32, got {tensor.dtype}")

        if not torch.isfinite(tensor).all():
            raise ValueError(f"{processed_path}: tensor contains NaN or Inf")

        min_value = float(tensor.min())
        max_value = float(tensor.max())
        if min_value < -1e-6 or max_value > 1.0 + 1e-6:
            raise ValueError(
                f"{processed_path}: tensor values outside [0, 1], "
                f"min={min_value}, max={max_value}"
            )

        if label not in {0, 1, 2}:
            raise ValueError(f"{processed_path}: invalid model label {label}")
