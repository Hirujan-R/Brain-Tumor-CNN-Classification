from pathlib import Path


TARGET_HEIGHT = 224
TARGET_WIDTH = 224
TARGET_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, 3)

EXPECTED_SOURCE_SHAPE = (512, 512)

NORMALIZATION = "zscore"  # IMPORTANT: Actual preprocessing uses zscore (mean=0, std=1)
INTERPOLATION = "bilinear"
OUTPUT_DTYPE = "float32"
CHANNELS = 3

N_FOLDS = 5

INDEX_PATH = Path("data/ingested/index.csv")
SPLIT_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/processed")
IMAGES_DIR = OUTPUT_DIR / "images"
PROCESSED_INDEX_PATH = OUTPUT_DIR / "processed_index.csv"
PREPROCESSING_STATS_PATH = OUTPUT_DIR / "preprocessing_stats.json"
PREPROCESSING_REPORT_PATH = OUTPUT_DIR / "preprocessing_report.json"

FILE_NAME_WIDTH = 6
CLEAN_OUTPUT = True
