import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

class Evaluator:
    """
    Handles evaluation of PyTorch models over validation or test DataLoaders.
    Calculates moving average loss and multi-class accuracy.
    """
    def __init__(self, model: nn.Module, criterion: nn.Module, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Runs inference on the provided dataloader.
        
        Returns:
            metrics (Dict): Dictionary containing 'loss' and 'accuracy'.
            all_preds (np.ndarray): Flattened array of all predicted classes.
            all_targets (np.ndarray): Flattened array of all true labels.
        """
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle possible metadata returning from BrainTumorDataset
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # GoogLeNet eval mode returns just logits, but in case it returns namedtuple, handle it
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                # Compute loss
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item() * images.size(0)
                
                # Compute accuracy
                _, preds = torch.max(logits, 1)
                correct_predictions += torch.sum(preds == labels).item()
                total_samples += images.size(0)
                
                # Collect for downstream metrics (e.g., F1, Confusion Matrix)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                
        avg_loss = total_loss / total_samples
        
        preds_arr = np.concatenate(all_preds)
        targets_arr = np.concatenate(all_targets)
        probs_arr = np.concatenate(all_probs)

        accuracy = balanced_accuracy_score(targets_arr, preds_arr)
        
        f1 = f1_score(targets_arr, preds_arr, average='macro')
        precision = precision_score(targets_arr, preds_arr, average="weighted")
        recall = recall_score(targets_arr, preds_arr, average="weighted")
        roc_auc = roc_auc_score(
            targets_arr,
            probs_arr,
            multi_class="ovr",
            average="weighted"
        )
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }
        
        return metrics, preds_arr, targets_arr
