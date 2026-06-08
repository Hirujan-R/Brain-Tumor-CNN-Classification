from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


class ImageLoadError(RuntimeError):
    pass


def load_mat_image(filepath: str) -> np.ndarray:
    """
    Load cjdata.image from a MATLAB v7.3 HDF5 .mat file.

    The transpose matches the ingestion parser and restores the image to the
    displayed 512x512 orientation used throughout the project registry.
    """
    if h5py is None:
        raise ImageLoadError(
            "h5py is required to load MATLAB v7.3 .mat files. "
            "Install project requirements or run inside the project environment."
        )

    path = Path(filepath)
    if not path.exists():
        raise ImageLoadError(f"Missing .mat file: {filepath}")

    try:
        with h5py.File(path, "r") as handle:
            if "cjdata" not in handle:
                raise ImageLoadError(f"Missing cjdata group: {filepath}")

            cjdata = handle["cjdata"]
            if "image" not in cjdata:
                raise ImageLoadError(f"Missing cjdata.image dataset: {filepath}")

            image = np.asarray(cjdata["image"]).T
    except OSError as exc:
        raise ImageLoadError(f"Could not read .mat file {filepath}: {exc}") from exc

    return image
