import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from src.inference.predict import predict, preprocess_raw_image

class DummyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    
    def forward(self, x):
        # Return dummy logits for 3 classes based on batch size
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 3)
        # Force class 2 to have highest logit for class prediction testing
        logits[:, 2] = 10.0
        return logits

class TestPredictPipeline(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()

    def test_predict_shape_hwc(self):
        # Input shape (H, W, C) = (224, 224, 3)
        img = np.random.rand(224, 224, 3).astype(np.float32)
        res = predict(self.model, img, device="cpu", preprocess_raw=False)
        
        self.assertEqual(res["predicted_label"], 2)
        self.assertEqual(res["original_label"], 3)  # Model label 2 -> Original label 3 (pituitary)
        self.assertEqual(res["class_name"], "pituitary")
        self.assertEqual(len(res["probabilities"]), 3)
        self.assertAlmostEqual(res["probabilities"][2], 1.0, places=3)

    def test_predict_shape_chw(self):
        # Input shape (C, H, W) = (3, 224, 224)
        img = np.random.rand(3, 224, 224).astype(np.float32)
        res = predict(self.model, img, device="cpu", preprocess_raw=False)
        
        self.assertEqual(res["predicted_label"], 2)
        self.assertEqual(res["original_label"], 3)
        self.assertEqual(res["class_name"], "pituitary")

    def test_preprocess_raw_image_clipping(self):
        # Test raw image preprocessing with and without clipping compatibility
        raw_img = np.random.randint(0, 256, size=(512, 512)).astype(np.float32)
        
        # 1. With clipping (compatibility mode)
        processed_clipped = preprocess_raw_image(raw_img, clip_compatibility=True)
        self.assertEqual(processed_clipped.shape, (224, 224, 3))
        self.assertTrue((processed_clipped >= 0.0).all() and (processed_clipped <= 1.0).all())
        
        # 2. Without clipping (raw z-score range preserved)
        processed_unclipped = preprocess_raw_image(raw_img, clip_compatibility=False)
        self.assertEqual(processed_unclipped.shape, (224, 224, 3))
        # Z-score normalization generates negative values, which should not be clipped to 0
        has_negatives = (processed_unclipped < 0.0).any()
        self.assertTrue(has_negatives, "Z-score normalized image without clipping should contain negative values.")

    def test_predict_with_raw_preprocessing(self):
        # Pass a raw 2D numpy array and select preprocess_raw=True
        raw_img = np.random.randint(0, 256, size=(512, 512)).astype(np.float32)
        res = predict(self.model, raw_img, device="cpu", preprocess_raw=True)
        
        self.assertEqual(res["predicted_label"], 2)
        self.assertEqual(res["original_label"], 3)
        self.assertEqual(res["class_name"], "pituitary")

    def test_predict_invalid_shape(self):
        # Pass an array with invalid dimensions
        img = np.random.rand(224).astype(np.float32)
        with self.assertRaises(ValueError):
            predict(self.model, img, device="cpu")

if __name__ == "__main__":
    unittest.main()
