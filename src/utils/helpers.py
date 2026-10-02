import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Any, path: str | Path, indent: int = 4) -> None:
    """Serialize a payload to JSON, creating parent directories as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=indent, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def _json_default(value):
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
