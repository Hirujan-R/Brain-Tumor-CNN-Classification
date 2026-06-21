"""
Retrain the model with detailed diagnostics to understand what's happening.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from collections import Counter

from src.datasets.brain_tumor_dataset import BrainTumorDataset
from src.datasets.dataset_utils import load_processed_registry, make_dataloader
from src.models.googlenet import GoogLeNetBrainTumor
from src.training.loss import get_weighted_loss
from src.training.trainer import Trainer

def count_labels_in_loader(loader, name="DataLoader"):
    """Count labels in a dataloader to verify class distribution."""
    print(f"\nCounting labels in {name}...")
    all_labels = []
    for batch in loader:
        if len(batch) == 3:
            _, labels, _ = batch
        else:
            _, labels = batch
        all_labels.extend(labels.tolist())
    
    counter = Counter(all_labels)
    total = len(all_labels)
    print(f"  Total samples: {total}")
    class_names = {0: 'glioma', 1: 'meningioma', 2: 'pituitary'}
    for label in sorted(counter.keys()):
        count = counter[label]
        pct = count / total * 100
        print(f"  {class_names[label]} (class {label}): {count} ({pct:.1f}%)")
    
    return counter

def main():
    print("="*80)
    print("RETRAINING WITH DIAGNOSTICS")
    print("="*80)
    
    # Configuration
    epochs = 30
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nConfiguration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Device: {device}")
    
    # Load data with proper split
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    processed_index_path = "data/processed/processed_index.csv"
    registry_df = load_processed_registry(processed_index_path)
    
    # CRITICAL: Use fold-based split
    train_df = registry_df[registry_df['fold'] != 0].reset_index(drop=True)
    val_df = registry_df[registry_df['fold'] == 0].reset_index(drop=True)
    
    print(f"\nDataset split:")
    print(f"  Total samples: {len(registry_df)}")
    print(f"  Training samples: {len(train_df)} ({len(train_df)/len(registry_df)*100:.1f}%)")
    print(f"  Validation samples: {len(val_df)} ({len(val_df)/len(registry_df)*100:.1f}%)")
    
    # Check class distribution
    from src.datasets.label_mapping import encode_label
    
    print(f"\nTraining set class distribution:")
    train_labels = train_df['label'].apply(encode_label)
    train_dist = train_labels.value_counts().sort_index()
    class_names = {0: 'glioma', 1: 'meningioma', 2: 'pituitary'}
    for label, count in train_dist.items():
        pct = count / len(train_df) * 100
        print(f"  {class_names[label]} (class {label}): {count} ({pct:.1f}%)")
    
    print(f"\nValidation set class distribution:")
    val_labels = val_df['label'].apply(encode_label)
    val_dist = val_labels.value_counts().sort_index()
    for label, count in val_dist.items():
        pct = count / len(val_df) * 100
        print(f"  {class_names[label]} (class {label}): {count} ({pct:.1f}%)")
    
    # Create datasets
    train_ds = BrainTumorDataset(
        processed_index_path=processed_index_path,
        registry_df=train_df,
        return_metadata=False,
        validate_files=False  # Skip validation for speed
    )
    
    val_ds = BrainTumorDataset(
        processed_index_path=processed_index_path,
        registry_df=val_df,
        return_metadata=False,
        validate_files=False
    )
    
    # Create dataloaders
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    
    # DIAGNOSTIC: Count actual labels in first epoch of dataloader
    print("\n" + "="*80)
    print("DIAGNOSTIC: Checking DataLoader class distribution")
    print("="*80)
    train_counter = count_labels_in_loader(train_loader, "Training DataLoader")
    val_counter = count_labels_in_loader(val_loader, "Validation DataLoader")
    
    # Verify distributions match
    print("\n✓ Data loading verification complete")
    if len(train_counter) == 3 and len(val_counter) == 3:
        print("  All 3 classes present in both train and val loaders")
    else:
        print("  ⚠ WARNING: Not all classes present!")
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = GoogLeNetBrainTumor(num_classes=3, pretrained=True, aux_logits=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create loss function with class weights
    print("\n" + "="*80)
    print("LOSS FUNCTION")
    print("="*80)
    
    def compute_class_counts(train_dataset):
        counts = {0: 0, 1: 0, 2: 0}
        for idx in range(len(train_dataset)):
            row = train_dataset.registry.iloc[idx]
            label = encode_label(row['label'])
            counts[label] += 1
        return counts
    
    class_counts = compute_class_counts(train_ds)
    print(f"  Class counts: {class_counts}")
    
    criterion = get_weighted_loss(class_counts, device=device)
    
    # Calculate and display weights
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = []
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        w = total_samples / (num_classes * count)
        weights.append(w)
    
    print(f"  Class weights:")
    for cls, weight in enumerate(weights):
        print(f"    {class_names[cls]} (class {cls}): {weight:.4f}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device
    )
    
    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Watch for:")
    print(f"  1. Training loss decreasing")
    print(f"  2. Validation loss decreasing (but DIFFERENT from training loss)")
    print(f"  3. Validation accuracy increasing")
    print(f"  4. All 3 classes being predicted")
    print()
    
    history = trainer.fit(num_epochs=epochs)
    
    # Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    model_path = "model/model_diagnostic.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  ✓ Model saved to: {model_path}")
    
    # Also save to /tmp for compatibility
    tmp_path = "/tmp/model.pth"
    torch.save(model.state_dict(), tmp_path)
    print(f"  ✓ Model saved to: {tmp_path}")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    print(f"\nFinal metrics:")
    print(f"  Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final validation accuracy: {history['val_accuracy'][-1]*100:.2f}%")
    print(f"  Final validation F1: {history['val_f1'][-1]:.4f}")
    
    print(f"\nBest metrics:")
    best_val_acc_idx = np.argmax(history['val_accuracy'])
    print(f"  Best validation accuracy: {history['val_accuracy'][best_val_acc_idx]*100:.2f}% (epoch {best_val_acc_idx+1})")
    print(f"  Best validation F1: {max(history['val_f1']):.4f}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Copy the model:")
    print("   cp model/model_diagnostic.pth model/model.pth")
    print("\n2. Verify performance:")
    print("   python3 -m src.inference.verify_train \\")
    print("       --model model/model.pth \\")
    print("       --index data/processed/processed_index.csv \\")
    print("       --out mismatches_new.csv")
    print()

if __name__ == "__main__":
    main()
