import zipfile
from pathlib import Path

zip_dir = Path("data/raw/zips")
extract_dir = Path("data/interim/extracted_mat")

extract_dir.mkdir(parents=True, exist_ok=True)

for zip_file in zip_dir.glob("*.zip"):
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(extract_dir)