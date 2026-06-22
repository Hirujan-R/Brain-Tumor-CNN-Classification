import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Dict, Any, Optional
import io

try:
    from .evaluator import Evaluator
    from .checkpointing import ModelCheckpoint
except ImportError:
    from evaluator import Evaluator
    from checkpointing import ModelCheckpoint

class Trainer:
    """
    Main training loop orchestrator for PyTorch models.
    Handles epochs, optimization steps, scheduler updates, and coordinates evaluation and checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        criterion: nn.Module,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        
        self.evaluator = Evaluator(model, criterion, device)
        
        # We will save the best model based on validation loss by default
        self.checkpointer = ModelCheckpoint(
            filepath=f"best_model.pth",
            monitor='val_loss',
            mode='min'
        )

    def train_epoch(self, epoch: int = 0) -> float:
        """Runs a single epoch of training."""
        if hasattr(self, 'epoch_callback') and self.epoch_callback:
            self.epoch_callback(epoch)
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Handle auxiliary logits for models like GoogLeNet in train mode
            if isinstance(outputs, tuple) and len(outputs) == 3:
                logits, aux2, aux1 = outputs
                loss1 = self.criterion(logits, labels)
                loss2 = self.criterion(aux2, labels)
                loss3 = self.criterion(aux1, labels)
                # GoogLeNet standard auxiliary loss weighting
                loss = loss1 + 0.3 * loss2 + 0.3 * loss3
            elif hasattr(outputs, 'logits'):
                loss = self.criterion(outputs.logits, labels)
            else:
                loss = self.criterion(outputs, labels)
                
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
        avg_loss = total_loss / total_samples
        return avg_loss

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        """
        Runs the full training process for the specified number of epochs.
        
        Returns:
            history (Dict): Dictionary tracking training and validation metrics over epochs.
        """
        import mlflow

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_roc_auc': [],
            'lr': [],
            'best_preds': None,
            'best_targets': None
        }
        
        print(f"Starting training for {num_epochs} epochs on device: {self.device}")
        

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch=epoch)
            
            # Evaluate
            val_metrics, preds_arr, targets_arr = self.evaluator.evaluate(self.val_loader)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1']
            val_precision = val_metrics['precision']
            val_recall = val_metrics['recall']
            val_roc_auc = val_metrics['roc_auc']
            
            # Step the scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val Acc: {val_acc*100:.2f}% - "
                  f"Val F1: {val_f1:.4f} - "
                  f"LR: {current_lr:.6f}")
                  
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_roc_auc'].append(val_roc_auc)
            history['lr'].append(current_lr)
            
            # MLflow logging per epoch
            if mlflow.active_run():
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_roc_auc": val_roc_auc,
                    "lr": current_lr
                }, step=epoch)
            
            # Checkpoint
            is_best = self.checkpointer.step(
                current_metric=val_loss,
                model=self.model,
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )
            
            if is_best:
                history["best_preds"] = preds_arr
                history["best_targets"] = targets_arr
            
        print("Training completed.")
            
        return history
