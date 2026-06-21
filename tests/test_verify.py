import unittest
import pandas as pd
import os
import shutil
import tempfile
from src.inference.verify import verify_prediction, verify_batch_predictions

# Define mock registry data
MOCK_DATA = {
    "image_id": [1, 2, 3, 10],
    "processed_path": [
        "data/processed/images/000001.npy",
        "data/processed/images/000002.npy",
        "data/processed/images/000003.npy",
        "data/processed/images/000010.npy"
    ],
    "original_path": [
        "/Users/Hirujan/Documents/GitHub/Brain-Tumor-CNN-Classification/data/interim/extracted_mat/1.mat",
        "/Users/Hirujan/Documents/GitHub/Brain-Tumor-CNN-Classification/data/interim/extracted_mat/2.mat",
        "/Users/Hirujan/Documents/GitHub/Brain-Tumor-CNN-Classification/data/interim/extracted_mat/3.mat",
        "/Users/Hirujan/Documents/GitHub/Brain-Tumor-CNN-Classification/data/interim/extracted_mat/10.mat"
    ],
    "patient_id": [1001, 1002, 1003, 1004],
    "label": [2, 1, 3, 2],  # Figshare labels: 2 = glioma, 1 = meningioma, 3 = pituitary
    "height": [224, 224, 224, 224],
    "width": [224, 224, 224, 224],
    "channels": [3, 3, 3, 3],
    "fold": [1, 1, 2, 3]
}

class TestVerify(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and file
        self.test_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.test_dir, "processed_index.csv")
        df = pd.DataFrame(MOCK_DATA)
        df.to_csv(self.csv_file, index=False)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_verify_prediction_success_by_int_id(self):
        # Image 1 (label 2 -> glioma -> model label 0)
        # Correct prediction
        res = verify_prediction(pred_class=0, image_identifier=1, processed_index_path=self.csv_file)
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["image_id"], 1)
        self.assertEqual(res["predicted_label"], 0)
        self.assertEqual(res["predicted_class_name"], "glioma")
        self.assertEqual(res["true_label"], 0)
        self.assertEqual(res["true_class_name"], "glioma")

        # Incorrect prediction (image 1 predicts meningioma -> model label 1)
        res_incorrect = verify_prediction(pred_class=1, image_identifier=1, processed_index_path=self.csv_file)
        self.assertFalse(res_incorrect["is_correct"])
        self.assertEqual(res_incorrect["predicted_label"], 1)
        self.assertEqual(res_incorrect["true_label"], 0)

    def test_verify_prediction_success_by_string_id(self):
        # Image 2 (label 1 -> meningioma -> model label 1)
        res = verify_prediction(pred_class=1, image_identifier="2", processed_index_path=self.csv_file)
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["image_id"], 2)
        self.assertEqual(res["true_class_name"], "meningioma")

    def test_verify_prediction_success_by_processed_path(self):
        # Image 3 (label 3 -> pituitary -> model label 2)
        path = "data/processed/images/000003.npy"
        res = verify_prediction(pred_class=2, image_identifier=path, processed_index_path=self.csv_file)
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["image_id"], 3)
        self.assertEqual(res["true_class_name"], "pituitary")

    def test_verify_prediction_success_by_filename(self):
        # Image 10 (label 2 -> glioma -> model label 0)
        filename = "000010.npy"
        res = verify_prediction(pred_class=0, image_identifier=filename, processed_index_path=self.csv_file)
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["image_id"], 10)
        self.assertEqual(res["true_class_name"], "glioma")

    def test_verify_prediction_success_by_original_filename(self):
        # Image 2 (label 1 -> meningioma -> model label 1)
        filename = "2.mat"
        res = verify_prediction(pred_class=1, image_identifier=filename, processed_index_path=self.csv_file)
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["image_id"], 2)
        self.assertEqual(res["true_class_name"], "meningioma")

    def test_verify_prediction_missing_identifier(self):
        # Try looking up image 99 which does not exist
        with self.assertRaises(ValueError) as context:
            verify_prediction(pred_class=1, image_identifier=99, processed_index_path=self.csv_file)
        self.assertIn("Could not find image with identifier '99'", str(context.exception))

    def test_verify_prediction_invalid_pred_class(self):
        # Try passing an invalid pred class like 3
        with self.assertRaises(ValueError) as context:
            verify_prediction(pred_class=3, image_identifier=1, processed_index_path=self.csv_file)
        self.assertIn("Invalid predicted class label: 3", str(context.exception))

    def test_verify_prediction_missing_file(self):
        with self.assertRaises(FileNotFoundError) as context:
            verify_prediction(pred_class=1, image_identifier=1, processed_index_path="nonexistent_index.csv")
        self.assertIn("Processed index CSV not found", str(context.exception))

    def test_verify_batch_predictions(self):
        pred_classes = [0, 2, 1]
        image_identifiers = [1, "000002.npy", "data/processed/images/000003.npy"]

        # Image 1: true label glioma (0), pred: 0 -> correct
        # Image 2: true label meningioma (1), pred: 2 -> incorrect
        # Image 3: true label pituitary (2), pred: 1 -> incorrect
        res = verify_batch_predictions(pred_classes, image_identifiers, processed_index_path=self.csv_file)

        self.assertAlmostEqual(res["accuracy"], 1.0 / 3.0)
        self.assertEqual(res["correct_count"], 1)
        self.assertEqual(res["total_count"], 3)
        self.assertEqual(len(res["details"]), 3)
        self.assertTrue(res["details"][0]["is_correct"])
        self.assertFalse(res["details"][1]["is_correct"])
        self.assertFalse(res["details"][2]["is_correct"])

    def test_verify_batch_predictions_length_mismatch(self):
        pred_classes = [0, 1]
        image_identifiers = [1]
        with self.assertRaises(ValueError) as context:
            verify_batch_predictions(pred_classes, image_identifiers, processed_index_path=self.csv_file)
        self.assertIn("Length mismatch", str(context.exception))

if __name__ == "__main__":
    unittest.main()
