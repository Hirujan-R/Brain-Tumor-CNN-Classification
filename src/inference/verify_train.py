"""verify_train.py

Verify model predictions over the full processed dataset used for training.
Generates a summary (accuracy, confusion matrix, classification report)
and optionally writes mismatched samples to CSV for inspection.

Usage:
    python -m src.inference.verify_train --model model/model.pth --index data/processed/processed_index.csv --out mismatches.csv
"""
import argparse
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from src.inference.predict import load_model, predict
from src.datasets.label_mapping import encode_label, MODEL_LABEL_TO_CLASS_NAME


def verify_all(model_path: str, processed_index_path: str, device: str = "cpu", out_csv: str | None = None):
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device=device)

    df = pd.read_csv(processed_index_path)
    n = len(df)
    y_true = []
    y_pred = []
    mismatches = []

    for _, row in tqdm(df.iterrows(), total=n, desc="Verifying dataset predictions"):
        path = row["processed_path"]
        image_id = int(row["image_id"]) if "image_id" in row else path
        true_label_original = int(row["label"])  # original Figshare label (1,2,3)
        true_label = encode_label(true_label_original)

        img = np.load(path)
        # predict() returns (pred_class, probs) in the current implementation
        pred_class, probs = predict(model, img, device=device)
        pred = int(pred_class)

        y_true.append(true_label)
        y_pred.append(pred)

        if pred != true_label:
            mismatches.append({
                "image_id": image_id,
                "processed_path": path,
                "true_label": true_label,
                "predicted_label": pred,
                "predicted_class_name": MODEL_LABEL_TO_CLASS_NAME.get(pred),
                "probabilities": list(map(float, np.asarray(probs).tolist())),
            })

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    print(f"Total: {n} | Accuracy: {acc*100:.2f}%")

    labels = sorted(list(MODEL_LABEL_TO_CLASS_NAME.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=[MODEL_LABEL_TO_CLASS_NAME[l] for l in labels]))

    if out_csv:
        keys = ["image_id", "processed_path", "true_label", "predicted_label", "predicted_class_name", "probabilities"]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in mismatches:
                writer.writerow(r)
        print(f"Wrote {len(mismatches)} mismatches to: {out_csv}")

    return {
        "total": n,
        "accuracy": acc,
        "confusion_matrix": cm,
        "mismatches_count": len(mismatches),
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify model predictions over processed dataset")
    parser.add_argument("--model", type=str, default="model/model.pth", help="Path to model checkpoint")
    parser.add_argument("--index", type=str, default="data/processed/processed_index.csv", help="Path to processed index CSV")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--out", type=str, default=None, help="CSV path to write mismatches")

    args = parser.parse_args()

    verify_all(model_path=args.model, processed_index_path=args.index, device=args.device, out_csv=args.out)


if __name__ == "__main__":
    main()
