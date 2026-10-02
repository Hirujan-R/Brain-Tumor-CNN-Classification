import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

try:
    from .evaluator import Evaluator
    from .checkpointing import ModelCheckpoint
except ImportError:
    from evaluator import Evaluator
    from checkpointing import ModelCheckpoint


class Trainer:
    """
    Main training loop orchestrator for PyTorch models.
    Handles epochs, optimization steps, scheduler updates, and coordinates
    evaluation and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.evaluator = Evaluator(model, criterion, device)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

        # Save the best model based on validation loss by default
        self.checkpointer = ModelCheckpoint(
            filepath=self.best_model_path,
            monitor=monitor,
            mode=mode,
        )

        self.epoch_callback = None

    def train_epoch(self, epoch: int = 0) -> float:
        """Runs a single epoch of training."""
        if self.epoch_callback is not None:
            self.epoch_callback(epoch)

        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            # Handle auxiliary logits for models like GoogLeNet in train mode
            if isinstance(outputs, tuple) and len(outputs) == 3:
                logits, aux2, aux1 = outputs
                loss1 = self.criterion(logits, labels)
                loss2 = self.criterion(aux2, labels)
                loss3 = self.criterion(aux1, labels)
                loss = loss1 + 0.3 * loss2 + 0.3 * loss3
            elif hasattr(outputs, "logits"):
                loss = self.criterion(outputs.logits, labels)
            else:
                loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        return total_loss / max(total_samples, 1)

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        """
        Runs the full training process for the specified number of epochs.

        Returns:
            history (Dict): Dictionary tracking training and validation metrics.
        """
        import mlflow

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_balanced_accuracy": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "val_roc_auc": [],
            "lr": [],
            "best_epoch": None,
            "best_val_loss": None,
            "best_preds": None,
            "best_targets": None,
        }

        print(f"Starting training for {num_epochs} epochs on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch=epoch)

            val_metrics, preds_arr, targets_arr = self.evaluator.evaluate(self.val_loader)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_bal_acc = val_metrics["balanced_accuracy"]
            val_f1 = val_metrics["f1"]
            val_precision = val_metrics["precision"]
            val_recall = val_metrics["recall"]
            val_roc_auc = val_metrics["roc_auc"]

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc * 100:.2f}% - "
                f"Val F1: {val_f1:.4f} - "
                f"LR: {current_lr:.6f}"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_balanced_accuracy"].append(val_bal_acc)
            history["val_f1"].append(val_f1)
            history["val_precision"].append(val_precision)
            history["val_recall"].append(val_recall)
            history["val_roc_auc"].append(val_roc_auc)
            history["lr"].append(current_lr)

            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_balanced_accuracy": val_bal_acc,
                        "val_f1": val_f1,
                        "val_precision": val_precision,
                        "val_recall": val_recall,
                        "val_roc_auc": val_roc_auc,
                        "lr": current_lr,
                    },
                    step=epoch,
                )

            is_best = self.checkpointer.step(
                current_metric=val_loss,
                model=self.model,
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                save_optimizer=False,
            )

            if is_best:
                history["best_epoch"] = epoch
                history["best_val_loss"] = val_loss
                history["best_preds"] = preds_arr
                history["best_targets"] = targets_arr

        print("Training completed.")
        return history
