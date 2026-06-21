import os
import pandas as pd
from typing import Union, List, Dict, Any
from src.datasets.label_mapping import encode_label, decode_label

def verify_prediction(
    pred_class: int,
    image_identifier: Union[int, str],
    processed_index_path: str = "data/processed/processed_index.csv"
) -> Dict[str, Any]:
    """
    Verifies if a single prediction made by the model is correct by comparing it
    against the ground truth label in the processed index CSV.

    Args:
        pred_class (int): Predicted model label (0 = glioma, 1 = meningioma, 2 = pituitary).
        image_identifier (Union[int, str]): Image ID (integer, or string representing integer),
            processed path, or filename of the image.
        processed_index_path (str): Path to the processed index CSV file.

    Returns:
        Dict[str, Any]: A dictionary containing verification details:
            - "is_correct" (bool): True if prediction matches true label, False otherwise.
            - "image_id" (int): The image ID.
            - "predicted_label" (int): The predicted class label.
            - "predicted_class_name" (str): Class name of prediction.
            - "true_label" (int): The true class label in model label space.
            - "true_class_name" (str): Class name of true label.
            - "processed_path" (str): Path to the processed image file.

    Raises:
        FileNotFoundError: If processed_index_path does not exist.
        ValueError: If image_identifier cannot be found in the registry, or if pred_class is invalid.
    """
    if not os.path.exists(processed_index_path):
        raise FileNotFoundError(f"Processed index CSV not found at: {processed_index_path}")

    if pred_class not in (0, 1, 2):
        raise ValueError(f"Invalid predicted class label: {pred_class}. Must be 0, 1, or 2.")

    # Load registry
    df = pd.read_csv(processed_index_path)

    # Attempt to locate the row in the registry using the identifier
    row = None

    # 1. Try matching by integer image_id
    try:
        img_id_int = int(image_identifier)
        matched_rows = df[df["image_id"] == img_id_int]
        if not matched_rows.empty:
            row = matched_rows.iloc[0]
    except (ValueError, TypeError):
        pass

    # 2. Try matching by exact processed_path or original_path if not matched yet
    if row is None and isinstance(image_identifier, str):
        matched_rows = df[(df["processed_path"] == image_identifier) | (df["original_path"] == image_identifier)]
        if not matched_rows.empty:
            row = matched_rows.iloc[0]

    # 3. Try matching by basename of processed_path or original_path (e.g. "000001.npy")
    if row is None and isinstance(image_identifier, str):
        identifier_basename = os.path.basename(image_identifier)
        # Vectorized check for basename matching
        matched_rows = df[
            df["processed_path"].apply(os.path.basename) == identifier_basename
        ]
        if not matched_rows.empty:
            row = matched_rows.iloc[0]
        else:
            matched_rows = df[
                df["original_path"].apply(lambda p: os.path.basename(str(p)) if pd.notna(p) else "") == identifier_basename
            ]
            if not matched_rows.empty:
                row = matched_rows.iloc[0]

    if row is None:
        raise ValueError(f"Could not find image with identifier '{image_identifier}' in registry.")

    # Extract true label and convert to model label space
    original_label = int(row["label"])
    true_label = encode_label(original_label)

    is_correct = (pred_class == true_label)

    return {
        "is_correct": is_correct,
        "image_id": int(row["image_id"]),
        "predicted_label": pred_class,
        "predicted_class_name": decode_label(pred_class),
        "true_label": true_label,
        "true_class_name": decode_label(true_label),
        "processed_path": str(row["processed_path"])
    }

def verify_batch_predictions(
    pred_classes: List[int],
    image_identifiers: List[Union[int, str]],
    processed_index_path: str = "data/processed/processed_index.csv"
) -> Dict[str, Any]:
    """
    Verifies a list of predictions against ground truth labels in the processed index CSV.

    Args:
        pred_classes (List[int]): List of predicted model labels.
        image_identifiers (List[Union[int, str]]): List of corresponding image identifiers.
        processed_index_path (str): Path to the processed index CSV file.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "accuracy" (float): Accuracy of the predictions (0.0 to 1.0).
            - "correct_count" (int): Number of correct predictions.
            - "total_count" (int): Total number of predictions.
            - "details" (List[Dict[str, Any]]): List of detailed verification dictionaries for each sample.

    Raises:
        ValueError: If lengths of pred_classes and image_identifiers do not match.
    """
    if len(pred_classes) != len(image_identifiers):
        raise ValueError(
            f"Length mismatch: pred_classes ({len(pred_classes)}) and "
            f"image_identifiers ({len(image_identifiers)}) must be of the same length."
        )

    details = []
    correct_count = 0

    for pred, identifier in zip(pred_classes, image_identifiers):
        res = verify_prediction(pred, identifier, processed_index_path)
        details.append(res)
        if res["is_correct"]:
            correct_count += 1

    total_count = len(pred_classes)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "details": details
    }
    

def main():
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Verify model predictions against ground truth labels.")
    parser.add_argument("--pred", type=int, help="Predicted class label (0 = glioma, 1 = meningioma, 2 = pituitary).")
    parser.add_argument("--id", type=str, help="Image ID, processed path, or filename.")
    parser.add_argument("--index", type=str, default="data/processed/processed_index.csv", help="Path to processed index CSV.")

    args = parser.parse_args()

    # If no arguments provided, run a quick self-test / demo
    if args.pred is None or args.id is None:
        print("No prediction or ID provided. Running verification demo with image '000001.npy' and predicted label 1 (meningioma):")
        try:
            demo_res = verify_prediction(pred_class=1, image_identifier="000001.npy", processed_index_path=args.index)
            pprint.pprint(demo_res)
        except Exception as e:
            print(f"Demo error: {e}")
            print("\nUsage: python -m src.inference.verify --pred <0|1|2> --id <image_id_or_path>")
    else:
        try:
            res = verify_prediction(pred_class=args.pred, image_identifier=args.id, processed_index_path=args.index)
            pprint.pprint(res)
        except Exception as e:
            print(f"Error during verification: {e}")

if __name__ == "__main__":
    main()