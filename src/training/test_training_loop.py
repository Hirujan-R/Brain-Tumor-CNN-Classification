import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

try:
    from .loss import get_weighted_loss
    from .trainer import Trainer
except ImportError:
    from loss import get_weighted_loss
    from trainer import Trainer

# Assuming we can run the test standalone from project root using:
# python -m src.training.test_training_loop
import sys
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'models'))
from cnn_baseline import CNNBaseline

def test_training_engine():
    print("Testing Stage 7 & 8 components...")
    
    # 1. Setup dummy data
    print("Setting up dummy dataloaders...")
    num_samples = 32
    # Create random images and labels (0, 1, 2)
    train_images = torch.randn(num_samples, 3, 224, 224)
    train_labels = torch.randint(0, 3, (num_samples,))
    val_images = torch.randn(num_samples // 2, 3, 224, 224)
    val_labels = torch.randint(0, 3, (num_samples // 2,))
    
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=8, shuffle=False)
    
    # 2. Setup loss
    print("Setting up Weighted CrossEntropyLoss...")
    # Using the real dataset class counts for testing the math
    class_counts = {0: 1426, 1: 708, 2: 930}
    criterion = get_weighted_loss(class_counts=class_counts)
    print(f"Calculated Weights: {criterion.weight.data}")
    
    # 3. Setup Model, Optimizer, Scheduler
    print("Initializing Model, AdamW, and CosineAnnealingLR...")
    model = CNNBaseline(num_classes=3)
    device = torch.device('cpu') # use cpu for quick test
    
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # T_max is the maximum number of iterations. We'll run for 2 epochs.
    scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)
    
    # 4. Setup Trainer
    print("Initializing Trainer...")
    test_checkpoint_dir = "models/test_checkpoints"
    os.makedirs(test_checkpoint_dir, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_dir=test_checkpoint_dir
    )
    
    # 5. Run Fit
    print("Running Trainer.fit() for 2 epochs...")
    history = trainer.fit(num_epochs=2)
    
    # 6. Verify Checkpoint
    checkpoint_path = f"{test_checkpoint_dir}/best_model.pth"
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created!"
    
    # Check history structure
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2
    assert len(history['lr']) == 2
    
    print("\n[OK] Training Engine Test Passed Successfully!")
    
    # Cleanup test checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists(test_checkpoint_dir):
        os.rmdir(test_checkpoint_dir)

if __name__ == "__main__":
    test_training_engine()
