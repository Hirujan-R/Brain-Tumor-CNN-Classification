from pathlib import Path

import pandas as pd


REQUIRED_INDEX_COLUMNS = {"filepath", "label", "patient_id"}


def image_id_from_path(filepath: str) -> int:
    return int(Path(filepath).stem)


def load_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(index_path)
    missing = REQUIRED_INDEX_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Index missing required columns: {sorted(missing)}")
    return df


def build_fold_map(split_dir: Path, n_folds: int) -> dict[int, int]:
    """
    Build image_id -> fold from validation CSVs.

    Across five-fold CV, every image should appear as validation exactly once.
    That makes the validation CSVs the safest source for each sample's fold.
    """
    fold_map = {}
    duplicates = []

    for fold in range(n_folds):
        val_path = split_dir / f"fold_{fold}_val.csv"
        if not val_path.exists():
            raise FileNotFoundError(f"Missing validation split: {val_path}")

        val_df = pd.read_csv(val_path)
        if "filepath" not in val_df.columns:
            raise ValueError(f"{val_path} missing filepath column")

        for filepath in val_df["filepath"]:
            image_id = image_id_from_path(filepath)
            if image_id in fold_map:
                duplicates.append(
                    {
                        "image_id": image_id,
                        "previous_fold": fold_map[image_id],
                        "duplicate_fold": fold,
                    }
                )
            fold_map[image_id] = fold

    if duplicates:
        raise ValueError(f"Images assigned to multiple folds: {duplicates[:10]}")

    return fold_map


def attach_folds(index_df: pd.DataFrame, fold_map: dict[int, int]) -> pd.DataFrame:
    df = index_df.copy()
    df["image_id"] = df["filepath"].map(image_id_from_path)

    if df["image_id"].duplicated().any():
        duplicates = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"Duplicate image IDs in index: {duplicates[:10]}")

    df["fold"] = df["image_id"].map(fold_map)
    missing_folds = df[df["fold"].isna()]["image_id"].tolist()
    if missing_folds:
        raise ValueError(f"Images missing fold assignment: {missing_folds[:10]}")

    extra_fold_ids = sorted(set(fold_map) - set(df["image_id"]))
    if extra_fold_ids:
        raise ValueError(f"Fold CSVs reference unknown image IDs: {extra_fold_ids[:10]}")

    df["fold"] = df["fold"].astype(int)
    return df.sort_values("image_id").reset_index(drop=True)
