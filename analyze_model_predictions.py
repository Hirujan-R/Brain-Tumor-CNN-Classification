"""
Analyze the current model's prediction patterns to understand what went wrong.

This script will:
1. Check prediction confidence scores
2. Analyze which samples are being misclassified
3. Look for patterns in the errors
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_mismatches(mismatch_csv="mismatches.csv", index_csv="data/processed/processed_index.csv"):
    """Analyze the mismatch patterns."""
    
    # Load data
    mismatches = pd.read_csv(mismatch_csv)
    full_index = pd.read_csv(index_csv)
    
    print("="*60)
    print("ANALYZING MODEL PREDICTION PATTERNS")
    print("="*60)
    
    # Total samples
    total_samples = len(full_index)
    total_mismatches = len(mismatches)
    accuracy = (total_samples - total_mismatches) / total_samples
    
    print(f"\nOverall Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {total_samples - total_mismatches}")
    print(f"Incorrect predictions: {total_mismatches}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    # Check what the model predicts
    print("\n" + "-"*60)
    print("Model Prediction Distribution (on all samples):")
    print("-"*60)
    
    # We need to get all predictions, not just mismatches
    # For now, let's analyze the mismatches
    
    print("\nPredictions on MISMATCHED samples:")
    pred_dist = mismatches['predicted_label'].value_counts().sort_index()
    for pred_label, count in pred_dist.items():
        print(f"  Predicted as {pred_label}: {count} samples ({count/len(mismatches)*100:.1f}%)")
    
    # True label distribution in mismatches
    print("\nTrue labels of MISMATCHED samples:")
    true_dist = mismatches['true_label'].value_counts().sort_index()
    for true_label, count in true_dist.items():
        print(f"  True label {true_label}: {count} samples ({count/len(mismatches)*100:.1f}%)")
    
    # Confusion analysis
    print("\n" + "-"*60)
    print("Confusion Patterns (True -> Predicted):")
    print("-"*60)
    
    confusion_counts = defaultdict(lambda: defaultdict(int))
    for _, row in mismatches.iterrows():
        true_label = int(row['true_label'])
        pred_label = int(row['predicted_label'])
        confusion_counts[true_label][pred_label] += 1
    
    label_names = {0: "glioma", 1: "meningioma", 2: "pituitary"}
    
    for true_label in sorted(confusion_counts.keys()):
        print(f"\nTrue label {true_label} ({label_names[true_label]}):")
        for pred_label, count in sorted(confusion_counts[true_label].items()):
            print(f"  -> Predicted as {pred_label} ({label_names[pred_label]}): {count} times")
    
    # Confidence analysis
    print("\n" + "-"*60)
    print("Prediction Confidence Analysis:")
    print("-"*60)
    
    # Parse probabilities
    def parse_probs(prob_str):
        # Remove brackets and parse
        prob_str = prob_str.strip('[]')
        return [float(x) for x in prob_str.split(',')]
    
    mismatches['probs_parsed'] = mismatches['probabilities'].apply(parse_probs)
    
    # Get max confidence for each prediction
    mismatches['max_confidence'] = mismatches['probs_parsed'].apply(max)
    
    print(f"\nConfidence statistics for INCORRECT predictions:")
    print(f"  Mean confidence: {mismatches['max_confidence'].mean():.4f}")
    print(f"  Median confidence: {mismatches['max_confidence'].median():.4f}")
    print(f"  Min confidence: {mismatches['max_confidence'].min():.4f}")
    print(f"  Max confidence: {mismatches['max_confidence'].max():.4f}")
    
    # High confidence errors
    high_conf_errors = mismatches[mismatches['max_confidence'] > 0.9]
    print(f"\nHigh confidence errors (>90%): {len(high_conf_errors)} samples")
    if len(high_conf_errors) > 0:
        print(f"  These are the most problematic cases - model is very confident but wrong!")
    
    # Check if model always predicts class 0
    print("\n" + "-"*60)
    print("Checking for bias towards class 0 (glioma):")
    print("-"*60)
    
    # Count predictions of 0 in mismatches
    pred_0_count = (mismatches['predicted_label'] == 0).sum()
    print(f"Mismatches predicted as class 0: {pred_0_count}/{len(mismatches)} ({pred_0_count/len(mismatches)*100:.1f}%)")
    
    if pred_0_count / len(mismatches) > 0.9:
        print("⚠ WARNING: Model is heavily biased towards predicting class 0!")
        print("  This is consistent with the classification report showing:")
        print("  - glioma: precision 0.47, recall 1.00")
        print("  - meningioma: precision 0.00, recall 0.00")
        print("  - pituitary: precision 0.00, recall 0.00")
    
    # Distribution by fold
    print("\n" + "-"*60)
    print("Error distribution by fold:")
    print("-"*60)
    
    # Merge with full index to get fold info
    mismatches_with_fold = mismatches.merge(
        full_index[['image_id', 'fold']], 
        on='image_id', 
        how='left'
    )
    
    fold_errors = mismatches_with_fold['fold'].value_counts().sort_index()
    for fold, count in fold_errors.items():
        # Count total in this fold
        total_in_fold = (full_index['fold'] == fold).sum()
        print(f"  Fold {fold}: {count}/{total_in_fold} errors ({count/total_in_fold*100:.1f}%)")


def plot_confidence_distribution(mismatch_csv="mismatches.csv"):
    """Plot the confidence distribution of wrong predictions."""
    
    mismatches = pd.read_csv(mismatch_csv)
    
    # Parse probabilities
    def parse_probs(prob_str):
        prob_str = prob_str.strip('[]')
        return [float(x) for x in prob_str.split(',')]
    
    mismatches['probs_parsed'] = mismatches['probabilities'].apply(parse_probs)
    mismatches['max_confidence'] = mismatches['probs_parsed'].apply(max)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(mismatches['max_confidence'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Number of Incorrect Predictions')
    plt.title('Confidence Distribution of Incorrect Predictions')
    plt.axvline(x=0.9, color='r', linestyle='--', label='90% confidence threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=150)
    print(f"\n✓ Saved confidence distribution plot to: confidence_distribution.png")


if __name__ == "__main__":
    analyze_mismatches()
    print("\n" + "="*60)
    plot_confidence_distribution()
