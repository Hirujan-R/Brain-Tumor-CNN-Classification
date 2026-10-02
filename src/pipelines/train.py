"""Unified training pipeline.

Runs 5-fold cross-validation for any of the supported models:

* Fine-tuned end-to-end CNNs:
    - cnn_baseline
    - googlenet
    - resnet18
    - vgg19
* Frozen pretrained CNN feature extraction + SVM:
    - googlenet+svm
    - resnet18+svm
    - vgg19+svm

Examples:
    python -m src.pipelines.train --model googlenet --epochs 30 --folds 5
    python -m src.pipelines.train --model resnet18+svm --folds 5
    python -m src.pipelines.train --model all
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from src.datasets.dataset_utils import create_fold_datasets, make_dataloader
from src.datasets.label_mapping import MODEL_LABEL_TO_CLASS_NAME, encode_label
from src.models.factory import (
    ALL_MODELS,
    build_classifier,
    build_feature_extractor,
    is_svm_model,
)
from src.models.transfer_models import SVMTrainerWrapper
from src.training.loss import get_weighted_loss
from src.training.trainer import Trainer
from src.utils.helpers import save_json
from src.utils.seed import set_seed

METRIC_KEYS = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the brain tumor training pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        default="googlenet",
        choices=list(ALL_MODELS) + ["all"],
        help="Model to train (or 'all' to train every model sequentially).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning LR.")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Head-only LR.")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=5,
        help="Epoch at which the full backbone is unfrozen (<=0 to train all weights from the start).",
    )
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds to run (1-5).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained weights (default: True).",
    )
    parser.add_argument(
        "--no-pretrained", action="store_false", dest="pretrained", help="Train from scratch."
    )
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-kernel", type=str, default="rbf", choices=["rbf", "linear", "poly", "sigmoid"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--processed-index",
        type=str,
        default="data/processed/processed_index.csv",
    )
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--mlflow", action="store_true", help="Log metrics to MLflow.")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (defaults to local ./mlruns).",
    )
    parser.add_argument("--experiment-name", type=str, default="Brain_Tumor_CV")
    return parser.parse_args()


def compute_class_counts(train_dataset) -> dict:
    """Count model-space class labels from the training registry."""
    counts = {0: 0, 1: 0, 2: 0}
    for original_label in train_dataset.registry["label"]:
        counts[encode_label(int(original_label))] += 1
    return counts


def build_train_transform():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
    )


def make_loaders(train_ds, val_ds, args, augment: bool):
    if augment:
        train_ds.transform = build_train_transform()
    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def run_fine_tune_fold(model_name, fold, args, device):
    train_ds, val_ds = create_fold_datasets(
        fold=fold,
        processed_index_path=args.processed_index,
        return_metadata=False,
    )
    train_loader, val_loader = make_loaders(train_ds, val_ds, args, augment=True)

    model = build_classifier(
        model_name,
        num_classes=3,
        pretrained=args.pretrained,
        aux_logits=True,
    ).to(device)

    class_counts = compute_class_counts(train_ds)
    criterion = get_weighted_loss(
        class_counts, device=device, label_smoothing=args.label_smoothing
    )

    freeze_first = args.unfreeze_epoch and args.unfreeze_epoch > 0
    if freeze_first:
        model.freeze_backbone()
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    else:
        model.unfreeze_all_layers()
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"{model_name}_fold{fold}")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_dir=ckpt_dir,
    )

    if freeze_first:

        def unfreeze_callback(epoch):
            if epoch == args.unfreeze_epoch:
                print("Unfreezing full model for fine-tuning...")
                model.unfreeze_all_layers()
                new_optimizer = AdamW(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
                optimizer.param_groups = new_optimizer.param_groups
                optimizer.state = new_optimizer.state

        trainer.epoch_callback = unfreeze_callback

    history = trainer.fit(num_epochs=args.epochs)

    best_idx = int(np.argmin(history["val_loss"]))
    metrics = {
        "accuracy": history["val_accuracy"][best_idx],
        "balanced_accuracy": history["val_balanced_accuracy"][best_idx],
        "f1": history["val_f1"][best_idx],
        "precision": history["val_precision"][best_idx],
        "recall": history["val_recall"][best_idx],
        "roc_auc": history["val_roc_auc"][best_idx],
    }
    info = {
        "fold": fold,
        "metrics": metrics,
        "best_epoch": int(best_idx + 1),
        "val_loss": float(history["val_loss"][best_idx]),
    }
    return info, history["best_targets"], history["best_preds"]


def run_svm_fold(model_name, fold, args, device):
    train_ds, val_ds = create_fold_datasets(
        fold=fold,
        processed_index_path=args.processed_index,
        return_metadata=False,
    )
    train_loader, val_loader = make_loaders(train_ds, val_ds, args, augment=False)

    extractor = build_feature_extractor(model_name).to(device)
    wrapper = SVMTrainerWrapper(
        extractor, kernel=args.svm_kernel, C=args.svm_c, random_state=args.seed
    )
    history = wrapper.fit(train_loader=train_loader, val_loader=val_loader)

    targets = history["targets"]
    preds = history["preds"]
    metrics = {
        "accuracy": float(history["val_accuracy"][0]),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "f1": float(history["val_f1"][0]),
        "precision": float(history["val_precision"][0]),
        "recall": float(history["val_recall"][0]),
        "roc_auc": float(history["val_roc_auc"][0]),
    }
    info = {"fold": fold, "metrics": metrics, "best_epoch": 1, "val_loss": None}
    return info, targets, preds


def aggregate_metrics(per_fold):
    aggregate = {}
    for key in METRIC_KEYS:
        values = [fold["metrics"][key] for fold in per_fold]
        aggregate[f"{key}_mean"] = float(np.nanmean(values))
        aggregate[f"{key}_std"] = float(np.nanstd(values))
    return aggregate


def log_to_mlflow(model_name, args, aggregate, per_fold):
    from src.mlops.mlflow_setup import setup_mlflow

    setup_mlflow(args.tracking_uri, args.experiment_name)
    import mlflow

    with mlflow.start_run(run_name=f"CV_{model_name}"):
        mlflow.log_params(
            {
                "model": model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "head_lr": args.head_lr,
                "weight_decay": args.weight_decay,
                "folds": args.folds,
                "pretrained": args.pretrained,
                "svm_c": args.svm_c,
                "seed": args.seed,
            }
        )
        mlflow.log_metrics(aggregate)
        for fold_info in per_fold:
            for key in METRIC_KEYS:
                mlflow.log_metric(
                    f"fold{fold_info['fold']}_{key}", fold_info["metrics"][key]
                )


def run_cv(model_name: str, args) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print(f"Model: {model_name} | Folds: {args.folds} | Epochs: {args.epochs}")
    print(f"Batch: {args.batch_size} | LR: {args.lr} | Device: {device}")
    print("=" * 60)

    svm = is_svm_model(model_name)
    per_fold = []
    all_targets = []
    all_preds = []

    for fold in range(args.folds):
        print(f"\n{'-' * 40}\nFold {fold}\n{'-' * 40}")
        if svm:
            info, targets, preds = run_svm_fold(model_name, fold, args, device)
        else:
            info, targets, preds = run_fine_tune_fold(model_name, fold, args, device)
        per_fold.append(info)
        all_targets.append(np.asarray(targets))
        all_preds.append(np.asarray(preds))
        print(
            f"Fold {fold} | Acc: {info['metrics']['accuracy'] * 100:.2f}% "
            f"| F1: {info['metrics']['f1']:.4f}"
        )

    aggregate = aggregate_metrics(per_fold)

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    labels = sorted(MODEL_LABEL_TO_CLASS_NAME.keys())
    cm = confusion_matrix(targets, preds, labels=labels)
    report = classification_report(
        targets,
        preds,
        labels=labels,
        target_names=[MODEL_LABEL_TO_CLASS_NAME[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "approach": "svm" if svm else "fine_tune",
        "num_folds": args.folds,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "pretrained": args.pretrained,
            "svm_c": args.svm_c,
            "svm_kernel": args.svm_kernel,
            "seed": args.seed,
        },
        "per_fold": per_fold,
        "aggregate": aggregate,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    output_path = Path(args.output_dir) / f"{model_name}_cv_metrics.json"
    save_json(summary, output_path)

    print(f"\n{'=' * 60}")
    print(f"Cross-validation complete for {model_name}")
    print(
        f"Accuracy: {aggregate['accuracy_mean'] * 100:.2f}% "
        f"+/- {aggregate['accuracy_std'] * 100:.2f}%"
    )
    print(
        f"F1 (macro): {aggregate['f1_mean']:.4f} +/- {aggregate['f1_std']:.4f}"
    )
    print(
        f"ROC-AUC: {aggregate['roc_auc_mean']:.4f} +/- {aggregate['roc_auc_std']:.4f}"
    )
    print(f"Saved metrics to: {output_path}")
    print(f"{'=' * 60}")

    if args.mlflow:
        try:
            log_to_mlflow(model_name, args, aggregate, per_fold)
            print("Logged run to MLflow.")
        except Exception as exc:  # pragma: no cover - optional dependency/server
            print(f"[WARN] MLflow logging failed: {exc}")

    return summary


def main():
    args = parse_args()
    models = list(ALL_MODELS) if args.model == "all" else [args.model]

    summaries = {}
    for model_name in models:
        summaries[model_name] = run_cv(model_name, args)

    if len(models) > 1:
        print("\n" + "=" * 60)
        print("ALL MODELS SUMMARY")
        print("=" * 60)
        for model_name, summary in summaries.items():
            acc = summary["aggregate"]["accuracy_mean"]
            f1 = summary["aggregate"]["f1_mean"]
            print(f"{model_name:>16} | Acc: {acc * 100:6.2f}% | F1: {f1:.4f}")


if __name__ == "__main__":
    main()
