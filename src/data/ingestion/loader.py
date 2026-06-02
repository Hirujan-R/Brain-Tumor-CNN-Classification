from pathlib import Path
from typing import List

def discover_mat_files(input_dir: str) -> List[str]:
    """
    Discover all .mat files in the dataset directory recursively.

    Parameters
    ----------
    input_dir : str
        Root directory containing extracted .mat files.

    Returns
    -------
    List[str]
        Sorted list of absolute file paths to .mat files.
    """

    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Recursively find all .mat files
    mat_files = list(input_path.rglob("*.mat"))

    from pathlib import Path
from typing import List


def discover_mat_files(input_dir: str) -> List[str]:
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    mat_files = list(input_path.rglob("*.mat"))

    if len(mat_files) == 0:
        raise ValueError(f"No .mat files found in: {input_dir}")

    # ----------------------------
    # NUMERIC SORT FIX
    # ----------------------------
    def extract_number(path: Path):
        # assumes filename like 1.mat, 10.mat, 100.mat
        return int(path.stem)

    mat_files = sorted(mat_files, key=extract_number)

    return [str(f.resolve()) for f in mat_files]