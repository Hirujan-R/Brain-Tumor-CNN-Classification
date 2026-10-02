from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    """Compute a standard set of multi-class classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "roc_auc": float("nan"),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if len(np.unique(y_true)) == y_prob.shape[1]:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics
