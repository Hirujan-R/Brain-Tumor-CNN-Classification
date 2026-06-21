import argparse
import os
import torch
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
from sklearn.model_selection import train_test_split
from src.models.cnn_baseline import CNNBaseline
from src.models.googlenet import GoogLeNetBrainTumor
from src.models.transfer_models import PretrainedFeatureExtractor, SVMTrainerWrapper
from src.training.loss import get_weighted_loss
from src.training.trainer import Trainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Complete Training Pipeline")
    parser.add_argument("--model", type=str, default="googlenet", choices=["cnn_baseline", "googlenet", "resnet18+svm", "googlenet+svm", "vgg19+svm"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def get_model(model_name: str, num_classes: int = 3):
    if model_name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes)
    elif model_name == "googlenet":
        return GoogLeNetBrainTumor(num_classes=num_classes, pretrained=True, aux_logits=True)
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
    model = get_model(args.model)

    if hasattr(model, "to"):
        model = model.to(device)

    # 1. Datasets
    processed_index_path = "data/processed/processed_index.csv"
    registry_df = load_processed_registry(processed_index_path)
    # Create a train/validation split from the processed registry to avoid
    # validating on the same data used for training (which would inflate metrics).
    train_df, val_df = train_test_split(
        registry_df,
        test_size=0.2,
        stratify=registry_df["label"],
        random_state=42,
    )

    train_ds = BrainTumorDataset(
        processed_index_path=processed_index_path,
        registry_df=train_df.reset_index(drop=True),
        return_metadata=False,
        validate_files=True,
    )

    val_ds = BrainTumorDataset(
        processed_index_path=processed_index_path,
        registry_df=val_df.reset_index(drop=True),
        return_metadata=False,
        validate_files=True,
    )

    train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    if args.model == "resnet18+svm" or args.model == "googlenet+svm" or args.model == "vgg19+svm":
        trainer = SVMTrainerWrapper(model)
        history = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    else:

        class_counts = compute_class_counts(train_ds)
        criterion = get_weighted_loss(class_counts, device=device)
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
    else:
        model_path = f"/tmp/model.pth"
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
