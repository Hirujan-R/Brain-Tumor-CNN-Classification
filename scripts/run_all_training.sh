#!/usr/bin/env bash
#
# Train every model (fine-tuned CNNs + pretrained CNN feature extractor + SVM)
# with 5-fold cross-validation and write a combined metrics table.
#
# Usage:
#   ./scripts/run_all_training.sh                     # train all models
#   ./scripts/run_all_training.sh googlenet resnet18  # train specific models
#   ./scripts/run_all_training.sh --setup             # install deps + dvc pull, then train all
#   ./scripts/run_all_training.sh --folds 1 --epochs 5
#   ./scripts/run_all_training.sh --models googlenet+svm --skip-setup
#
# All options can also be provided as environment variables:
#   FOLDS=5 EPOCHS=30 VGG_EPOCHS=20 BATCH_SIZE=32 NUM_WORKERS=2 DEVICE=cuda \
#   OUTPUT_DIR=reports MLFLOW_FLAG=--mlflow ./scripts/run_all_training.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-30}"
VGG_EPOCHS="${VGG_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-reports}"
MLFLOW_FLAG="${MLFLOW_FLAG:-}"
SKIP_SETUP=0
RUN_SETUP=0
PYTHON="${PYTHON:-python}"

ALL_MODELS=(
  cnn_baseline
  googlenet
  resnet18
  vgg19
  googlenet+svm
  resnet18+svm
  vgg19+svm
)

SELECTED_MODELS=()

usage() {
  cat <<'EOF'
Train every model (fine-tuned CNNs + pretrained CNN feature extractor + SVM)
with 5-fold cross-validation and write a combined metrics table.

Usage:
  ./scripts/run_all_training.sh                     # train all models
  ./scripts/run_all_training.sh googlenet resnet18  # train specific models
  ./scripts/run_all_training.sh --setup             # install deps + dvc pull, then train all
  ./scripts/run_all_training.sh --folds 1 --epochs 5
  ./scripts/run_all_training.sh --models googlenet+svm --skip-setup

All options can also be provided as environment variables:
  FOLDS=5 EPOCHS=30 VGG_EPOCHS=20 BATCH_SIZE=32 NUM_WORKERS=2 DEVICE=cuda \
  OUTPUT_DIR=reports MLFLOW_FLAG=--mlflow ./scripts/run_all_training.sh
EOF
  exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup) RUN_SETUP=1; shift ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --models) IFS=',' read -r -a SELECTED_MODELS <<< "$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --vgg-epochs) VGG_EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --mlflow) MLFLOW_FLAG="--mlflow"; shift ;;
    -h|--help) usage ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) SELECTED_MODELS+=("$1"); shift ;;
  esac
done

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
  SELECTED_MODELS=("${ALL_MODELS[@]}")
fi

# ---------------------------------------------------------------------------
# Move to repository root
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
echo "Repository: $REPO_ROOT"

# ---------------------------------------------------------------------------
# Optional setup: install dependencies + pull data
# ---------------------------------------------------------------------------
if [[ "$RUN_SETUP" -eq 1 && "$SKIP_SETUP" -eq 0 ]]; then
  echo "==> Installing Python dependencies"
  "$PYTHON" -m pip install -r requirements.txt

  echo "==> Pulling DVC data"
  "$PYTHON" -m dvc pull -v
fi

if [[ "$SKIP_SETUP" -eq 0 ]]; then
  if [[ ! -f "data/processed/processed_index.csv" ]]; then
    echo "==> Processed data missing; attempting 'python -m dvc pull'"
    if ! "$PYTHON" -m dvc pull -v; then
      echo "ERROR: 'python -m dvc pull' failed. Install dvc (pip install 'dvc[s3]')" >&2
      echo "       and configure AWS credentials, then re-run. Or pass --skip-setup" >&2
      echo "       if data/processed is already present." >&2
      exit 1
    fi
  fi

  if [[ ! -f "data/processed/processed_index.csv" ]]; then
    echo "ERROR: data/processed/processed_index.csv not found after dvc pull." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
epochs_for() {
  case "$1" in
    vgg19) echo "$VGG_EPOCHS" ;;
    *+svm) echo 0 ;;
    *) echo "$EPOCHS" ;;
  esac
}

device_args() {
  if [[ "$DEVICE" != "auto" ]]; then
    printf '%s' "--device $DEVICE"
  fi
}

mkdir -p "$OUTPUT_DIR"

echo
echo "Models:      ${SELECTED_MODELS[*]}"
echo "Folds:       $FOLDS"
echo "Fine-tune epochs: $EPOCHS (vgg19: $VGG_EPOCHS)"
echo "Batch size:  $BATCH_SIZE | Workers: $NUM_WORKERS | Device: $DEVICE"
echo "Output dir:  $OUTPUT_DIR"
echo

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for model in "${SELECTED_MODELS[@]}"; do
  echo "==================================================================="
  echo "TRAINING: $model"
  echo "==================================================================="

  common_args=(
    --model "$model"
    --folds "$FOLDS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
  )

  if [[ "$model" == *"+svm" ]]; then
    # SVM models ignore --epochs
    "$PYTHON" -m src.pipelines.train "${common_args[@]}" $(device_args) $MLFLOW_FLAG
  else
    "$PYTHON" -m src.pipelines.train \
      "${common_args[@]}" \
      --epochs "$(epochs_for "$model")" \
      $(device_args) $MLFLOW_FLAG
  fi
done

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
echo
echo "==> Aggregating metrics"
OUTPUT_DIR="$OUTPUT_DIR" "$PYTHON" - <<'PY'
import glob
import json
import os

import pandas as pd

output_dir = os.environ.get("OUTPUT_DIR", "reports")
rows = []
for path in sorted(glob.glob(os.path.join(output_dir, "*_cv_metrics.json"))):
    with open(path) as handle:
        payload = json.load(handle)
    agg = payload["aggregate"]
    rows.append({
        "model": payload["model"],
        "approach": payload["approach"],
        "folds": payload["num_folds"],
        "accuracy": agg["accuracy_mean"],
        "accuracy_std": agg["accuracy_std"],
        "balanced_accuracy": agg["balanced_accuracy_mean"],
        "precision": agg["precision_mean"],
        "recall": agg["recall_mean"],
        "f1": agg["f1_mean"],
        "roc_auc": agg["roc_auc_mean"],
    })

if not rows:
    raise SystemExit(f"No *_cv_metrics.json found in {output_dir}")

results = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
pd.set_option("display.float_format", lambda value: f"{value:.4f}")
print(results.to_string(index=False))

out_csv = os.path.join(output_dir, "all_models_metrics.csv")
results.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
PY

echo
echo "Done."
