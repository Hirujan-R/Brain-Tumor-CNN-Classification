import shutil
from pathlib import Path

import numpy as np

from src.preprocessing.config import FILE_NAME_WIDTH


def prepare_output_dirs(output_dir: Path, images_dir: Path, clean: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean and images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)


def processed_filename(image_id: int) -> str:
    return f"{int(image_id):0{FILE_NAME_WIDTH}d}.npy"


def processed_path(images_dir: Path, image_id: int) -> Path:
    return images_dir / processed_filename(image_id)


def save_processed_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, image, allow_pickle=False)
