import argparse
import os
import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import io

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
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross Validation Training Pipeline")
    parser.add_argument("--model", type=str, default="cnn_baseline", choices=["cnn_baseline", "googlenet", "resnet18+svm"])
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
    
    print(f"Starting 5-Fold Cross-Validation Pipeline")
    print(f"Model: {args.model} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Device: {device}")
    
    experiment = mlflow.set_experiment("Brain_Tumor_CV")
    
    oof_accuracies = []
    oof_f1_scores = []
    
    # Start Parent Run
    with mlflow.start_run(run_name=f"CV_{args.model}", experiment_id=experiment.experiment_id) as parent_run:
        mlflow.log_params(vars(args))
        
        for fold in range(5):
            print(f"\n{'='*40}")
            print(f"Starting Fold {fold}")
            print(f"{'='*40}")
            
            # Start Nested Child Run for the specific fold
            with mlflow.start_run(run_name=f"Fold_{fold}", nested=True, experiment_id=experiment.experiment_id):
                mlflow.log_param("fold", fold)
                
                # 1. Datasets
                train_ds, val_ds = create_fold_datasets(fold=fold, return_metadata=False)
                train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
                val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
                
                # 2. Weighted Loss
                class_counts = compute_class_counts(train_ds)
                criterion = get_weighted_loss(class_counts, device=device)
                
                # 3. Model, Optimizer, Scheduler
                model = get_model(args.model)

                if hasattr(model, "to"):
                    model = model.to(device)

                if args.model == "resnet18+svm":
                    trainer = SVMTrainerWrapper(model)
                    history = trainer.fit(train_loader=train_loader, val_loader=val_loader)
                else:
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
                

                # Best validation metrics (Trainer saves based on min val_loss)
                # Let's extract the validation accuracy and f1 at the epoch of min val_loss
                if history["val_loss"]:
                    best_idx = np.argmin(history["val_loss"])
                    best_targets = history["best_targets"]
                    best_preds = history["best_preds"]
                    best_val_acc = history["val_accuracy"][best_idx]
                    best_val_f1 = history["val_f1"][best_idx]
                else:
                    best_val_acc = history["val_accuracy"][0]
                    best_val_f1 = history["val_f1"][0]
                    best_targets = history["targets"]
                    best_preds = history["preds"]
                
                print(f"\nFold {fold} Best Val Acc: {best_val_acc*100:.2f}% | Best Val F1: {best_val_f1:.4f}")
                
                oof_accuracies.append(best_val_acc)
                oof_f1_scores.append(best_val_f1)
                
                mlflow.log_metrics({
                    "fold_best_val_accuracy": best_val_acc,
                    "fold_best_val_f1": best_val_f1
                })

                fig, ax = plt.subplots(figsize=(6, 5))
                cm = confusion_matrix(best_targets, best_preds)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Validation Confusion Matrix (Best Epoch)')
                # Save to a temp buffer and log as artifact in a fold-specific subfolder
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                plt.close(fig)

                # Write to a named temp file so mlflow.log_artifact can pick it up
                cm_path = f"/tmp/fold_{fold}_confusion_matrix.png"
                with open(cm_path, "wb") as f:
                    f.write(buf.read())

                # log_artifact with artifact_path keeps it inside the *current* (child) run's folder
                mlflow.log_artifact(cm_path, artifact_path=f"{args.model}_fold_{fold}")

                # --- Model: save as .pth and log to child run ---
                model_path = f"/tmp/fold_{fold}_model.pth"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path=f"{args.model}_fold_{fold}")
                
        # Aggregate OOF Performance
        mean_acc = np.mean(oof_accuracies)
        std_acc = np.std(oof_accuracies)
        mean_f1 = np.mean(oof_f1_scores)
        std_f1 = np.std(oof_f1_scores)
        
        print(f"\n{'='*40}")
        print("Cross-Validation Complete!")
        print(f"OOF Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"OOF F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"{'='*40}")

    
        # Log aggregated metrics to parent run
        mlflow.log_metrics({
            "oof_mean_accuracy": mean_acc,
            "oof_std_accuracy": std_acc,
            "oof_mean_f1": mean_f1,
            "oof_std_f1": std_f1
        })

if __name__ == "__main__":
    main()
