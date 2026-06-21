import argparse
import os
import torch
import torch.nn as nn
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import io
import joblib
from src.datasets.brain_tumor_dataset import BrainTumorDataset
from src.datasets.dataset_utils import load_processed_registry

# Update PYTHONPATH so we can run this from anywhere if needed, or assume it's run from root as `python -m src.pipelines.train_cv`
import sys
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)
if os.path.join(src_dir, 'models') not in sys.path:
    sys.path.append(os.path.join(src_dir, 'models'))

from src.datasets.dataset_utils import create_fold_datasets, make_dataloader
from src.models.cnn_baseline import CNNBaseline
from src.models.googlenet import GoogLeNetBrainTumor
from src.models.transfer_models import PretrainedFeatureExtractor, SVMTrainerWrapper
from src.training.loss import get_weighted_loss
from src.training.trainer import Trainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Complete Training Pipeline")
    parser.add_argument("--model", type=str, default="googlenet", choices=["cnn_baseline", "googlenet", "resnet18+svm", "googlenet+svm", "vgg19+svm"])
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights (default: True)")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="Train from scratch")
    return parser.parse_args()

def get_model(model_name: str, num_classes: int = 3, pretrained: bool = False):
    if model_name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes)
    elif model_name == "googlenet":
        return GoogLeNetBrainTumor(num_classes=num_classes, pretrained=pretrained, aux_logits=True)
    elif model_name == "googlenet+svm":
        extractor = PretrainedFeatureExtractor(base_model_name="googlenet")
        return extractor
    elif model_name == "vgg19+svm":
        extractor = PretrainedFeatureExtractor(base_model_name="vgg19")
        return extractor
    elif model_name == "resnet18+svm":
        extractor = PretrainedFeatureExtractor(base_model_name="resnet18")
        return extractor
    else:
        raise ValueError(f"Unknown model: {model_name}")

def compute_class_counts(train_dataset):
    """Utility to count classes in the training dataset to build Weighted Loss."""
    counts = {0: 0, 1: 0, 2: 0}
    # We can iterate through the registry to avoid loading all images
    for idx in range(len(train_dataset)):
        # Check label using registry directly to be fast
        row = train_dataset.registry.iloc[idx]
        from src.datasets.label_mapping import encode_label
        label = encode_label(row['label'])
        counts[label] += 1
    return counts

def main():
    
    args = parse_args()
    device = torch.device(args.device)
    
    print(f"Starting Complete Training Pipeline")
    print(f"Model: {args.model} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Device: {device}")
        
            
    # Start Nested Child Run for the specific fold
    # 3. Model, Optimizer, Scheduler
    model = get_model(args.model, pretrained=args.pretrained)

    if hasattr(model, "to"):
        model = model.to(device)

    # 1. Datasets - USE PROPER TRAIN/VAL SPLIT
    processed_index_path = "data/processed/processed_index.csv"
    registry_df = load_processed_registry(processed_index_path)
    
    # CRITICAL FIX: Use fold-based split instead of training on entire dataset
    # Use fold 0 for validation, rest for training
    train_df = registry_df[registry_df['fold'] != 0].reset_index(drop=True)
    val_df = registry_df[registry_df['fold'] == 0].reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)} | Validation samples: {len(val_df)}")
    print(f"Train class distribution:\n{train_df['label'].value_counts().sort_index()}")
    print(f"Val class distribution:\n{val_df['label'].value_counts().sort_index()}")
    
    train_ds = BrainTumorDataset(
        processed_index_path=processed_index_path,
        registry_df=train_df,
        return_metadata=False,
        validate_files=True
    )
    
    val_ds = BrainTumorDataset(
        processed_index_path=processed_index_path,
        registry_df=val_df,
        return_metadata=False,
        validate_files=True
    )

    # Add data augmentation for training (helps generalization with limited data)
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])
    
    train_ds.transform = train_transform
    train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    if args.model == "resnet18+svm" or args.model == "googlenet+svm" or args.model == "vgg19+svm":
        trainer = SVMTrainerWrapper(model)
        history = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    else:

        # Use standard CrossEntropyLoss (no class weights - they caused bias toward specific classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device
        )
                    
        
        history = trainer.fit(num_epochs=args.epochs)
            

    # --- Model: save as .pth and log to child run ---
    if args.model == "resnet18+svm" or args.model == "googlenet+svm" or args.model == "vgg19+svm":
        model_path = f"/tmp/svm.pkl"
        joblib.dump(trainer.svm, model_path)
        print(f"SVM model saved to: {model_path}")
    else:
        os.makedirs("model", exist_ok=True)
        model_path = "model/model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        # Also save to /tmp for backwards compatibility
        torch.save(model.state_dict(), "/tmp/model.pth")
        print(f"Model also saved to: /tmp/model.pth")


if __name__ == "__main__":
    main()