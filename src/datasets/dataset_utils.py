from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

try:
    from .brain_tumor_dataset import BrainTumorDataset
    from .label_mapping import ORIGINAL_LABEL_TO_CLASS_NAME
except ImportError:
    from brain_tumor_dataset import BrainTumorDataset
    from label_mapping import ORIGINAL_LABEL_TO_CLASS_NAME


def load_processed_registry(
    processed_index_path: str | Path = "data/processed/processed_index.csv",
) -> pd.DataFrame:
    return pd.read_csv(processed_index_path)


def filter_registry_by_fold(
    registry_df: pd.DataFrame,
    fold: int,
    split: str,
) -> pd.DataFrame:
    if split == "train":
        filtered = registry_df[registry_df["fold"] != fold]
    elif split in {"val", "validation"}:
        filtered = registry_df[registry_df["fold"] == fold]
    else:
        raise ValueError("split must be 'train' or 'val'")

    return filtered.reset_index(drop=True)


def create_fold_datasets(
    fold: int,
    processed_index_path: str | Path = "data/processed/processed_index.csv",
    return_metadata: bool = True,
    validate_files: bool = True,
) -> tuple[BrainTumorDataset, BrainTumorDataset]:
    registry_df = load_processed_registry(processed_index_path)
    train_df = filter_registry_by_fold(registry_df, fold=fold, split="train")
    val_df = filter_registry_by_fold(registry_df, fold=fold, split="val")

    train_dataset = BrainTumorDataset(
        processed_index_path=processed_index_path,
        registry_df=train_df,
        return_metadata=return_metadata,
        validate_files=validate_files,
    )
    val_dataset = BrainTumorDataset(
        processed_index_path=processed_index_path,
        registry_df=val_df,
        return_metadata=return_metadata,
        validate_files=validate_files,
    )
    return train_dataset, val_dataset


def make_dataloader(
    dataset: BrainTumorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    if num_workers == 0:
        persistent_workers = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def summarize_registry(registry_df: pd.DataFrame) -> dict:
    label_counts = registry_df["label"].value_counts().sort_index().to_dict()
    label_counts_named = {
        ORIGINAL_LABEL_TO_CLASS_NAME[int(label)]: int(count)
        for label, count in label_counts.items()
    }

    return {
        "samples": int(len(registry_df)),
        "patients": int(registry_df["patient_id"].nunique()),
        "class_counts": label_counts_named,
        "fold_counts": {
            str(int(fold)): int(count)
            for fold, count in registry_df["fold"].value_counts().sort_index().items()
        },
    }


def summarize_fold(
    fold: int,
    processed_index_path: str | Path = "data/processed/processed_index.csv",
) -> dict:
    registry_df = load_processed_registry(processed_index_path)
    train_df = filter_registry_by_fold(registry_df, fold=fold, split="train")
    val_df = filter_registry_by_fold(registry_df, fold=fold, split="val")

    return {
        "fold": int(fold),
        "train": summarize_registry(train_df),
        "validation": summarize_registry(val_df),
    }
