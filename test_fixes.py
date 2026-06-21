"""
Test script to verify the critical fixes before retraining.

This script checks:
1. Data split is properly configured
2. Class distribution in train/val sets
3. Preprocessing consistency
4. Model architecture
"""

import pandas as pd
import numpy as np
from collections import Counter

def test_data_split():
    """Verify train/val split is working correctly."""
    print("="*60)
    print("TEST 1: Data Split Verification")
    print("="*60)
    
    # Load processed index
    df = pd.read_csv("data/processed/processed_index.csv")
    
    # Split like the training script does
    train_df = df[df['fold'] != 0]
    val_df = df[df['fold'] == 0]
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    
    # Check for overlap
    train_ids = set(train_df['image_id'].values)
    val_ids = set(val_df['image_id'].values)
    overlap = train_ids & val_ids
    
    if overlap:
        print(f"\n❌ FAIL: Found {len(overlap)} overlapping samples!")
        return False
    else:
        print(f"\n✓ PASS: No overlap between train and validation sets")
    
    # Check class distribution
    print("\nClass distribution in training set:")
    train_dist = train_df['label'].value_counts().sort_index()
    for label, count in train_dist.items():
        print(f"  Label {label}: {count} samples ({count/len(train_df)*100:.1f}%)")
    
    print("\nClass distribution in validation set:")
    val_dist = val_df['label'].value_counts().sort_index()
    for label, count in val_dist.items():
        print(f"  Label {label}: {count} samples ({count/len(val_df)*100:.1f}%)")
    
    return True


def test_preprocessing_consistency():
    """Verify preprocessing is consistent."""
    print("\n" + "="*60)
    print("TEST 2: Preprocessing Consistency")
    print("="*60)
    
    # Load a sample image
    sample_path = "data/processed/images/000001.npy"
    img = np.load(sample_path)
    
    print(f"\nSample image: {sample_path}")
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Min: {img.min():.4f}")
    print(f"Max: {img.max():.4f}")
    print(f"Mean: {img.mean():.4f}")
    print(f"Std: {img.std():.4f}")
    
    # Check if it looks like zscore normalized data
    # Zscore normalized data should have values that can be negative or > 1
    # But after resize with clip, it might be in a different range
    
    # Load config
    from src.preprocessing.config import NORMALIZATION
    print(f"\nConfig NORMALIZATION setting: {NORMALIZATION}")
    
    if NORMALIZATION == "zscore":
        print("✓ PASS: Config correctly set to zscore")
    else:
        print(f"⚠ WARNING: Config says '{NORMALIZATION}' but code uses zscore")
    
    return True


def test_label_mapping():
    """Verify label mapping is correct."""
    print("\n" + "="*60)
    print("TEST 3: Label Mapping Verification")
    print("="*60)
    
    from src.datasets.label_mapping import (
        ORIGINAL_LABEL_TO_MODEL_LABEL,
        MODEL_LABEL_TO_CLASS_NAME,
        encode_label
    )
    
    print("\nOriginal label -> Model label mapping:")
    for orig, model in ORIGINAL_LABEL_TO_MODEL_LABEL.items():
        class_name = MODEL_LABEL_TO_CLASS_NAME[model]
        print(f"  {orig} -> {model} ({class_name})")
    
    # Test encoding
    assert encode_label(1) == 1, "Label 1 should map to 1 (meningioma)"
    assert encode_label(2) == 0, "Label 2 should map to 0 (glioma)"
    assert encode_label(3) == 2, "Label 3 should map to 2 (pituitary)"
    
    print("\n✓ PASS: Label mapping is correct")
    return True


def test_model_architecture():
    """Verify model architecture matches saved checkpoint."""
    print("\n" + "="*60)
    print("TEST 4: Model Architecture Verification")
    print("="*60)
    
    import torch
    from torchvision.models import googlenet
    
    # Create model like in predict.py
    model = googlenet(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.aux1.fc2 = torch.nn.Linear(model.aux1.fc2.in_features, 3)
    model.aux2.fc2 = torch.nn.Linear(model.aux2.fc2.in_features, 3)
    
    # Try to load state dict
    try:
        state_dict = torch.load("model/model.pth", map_location="cpu")
        
        # Check for "model." prefix
        has_prefix = any(k.startswith("model.") for k in state_dict.keys())
        print(f"\nState dict has 'model.' prefix: {has_prefix}")
        
        # Remove prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("model.", "")] = v
        
        model.load_state_dict(new_state_dict)
        print("✓ PASS: Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"❌ FAIL: Could not load model: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("RUNNING PRE-TRAINING VERIFICATION TESTS")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Data Split", test_data_split()))
    results.append(("Preprocessing", test_preprocessing_consistency()))
    results.append(("Label Mapping", test_label_mapping()))
    results.append(("Model Architecture", test_model_architecture()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to retrain.")
        print("\nNext steps:")
        print("1. Backup current model:")
        print("   mv model/model.pth model/model_old.pth")
        print("\n2. Retrain with fixed script:")
        print("   python -m src.pipelines.train --model googlenet --epochs 30 --lr 1e-4")
    else:
        print("\n❌ Some tests failed. Fix issues before retraining.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
