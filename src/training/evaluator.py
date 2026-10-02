from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    """
    Handles evaluation of PyTorch models over validation or test DataLoaders.
    Calculates loss, accuracy, balanced accuracy and multi-class metrics.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def evaluate(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Runs inference on the provided dataloader.

        Returns:
            metrics (Dict): loss, accuracy, balanced_accuracy, f1, precision,
                recall, roc_auc.
            all_preds (np.ndarray): Flattened array of predicted classes.
            all_targets (np.ndarray): Flattened array of true labels.
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                # GoogLeNet returns a namedtuple in train mode; unwrap defensively.
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

                loss = self.criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                _, preds = torch.max(logits, 1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_loss = total_loss / max(total_samples, 1)

        preds_arr = np.concatenate(all_preds)
        targets_arr = np.concatenate(all_targets)
        probs_arr = np.concatenate(all_probs)

        accuracy = accuracy_score(targets_arr, preds_arr)
        balanced_accuracy = balanced_accuracy_score(targets_arr, preds_arr)
        f1 = f1_score(targets_arr, preds_arr, average="macro", zero_division=0)
        precision = precision_score(
            targets_arr, preds_arr, average="weighted", zero_division=0
        )
        recall = recall_score(
            targets_arr, preds_arr, average="weighted", zero_division=0
        )

        roc_auc = float("nan")
        try:
            if len(np.unique(targets_arr)) == probs_arr.shape[1]:
                roc_auc = roc_auc_score(
                    targets_arr,
                    probs_arr,
                    multi_class="ovr",
                    average="weighted",
                )
        except ValueError:
            roc_auc = float("nan")

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }

        return metrics, preds_arr, targets_arr
