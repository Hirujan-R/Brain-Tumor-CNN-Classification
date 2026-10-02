from typing import Dict

import torch
import torch.nn as nn


def get_weighted_loss(
    class_counts: Dict[int, int],
    device: torch.device = torch.device("cpu"),
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """
    Creates a Weighted CrossEntropyLoss to handle class imbalance.
    Weights are calculated inversely proportional to class frequencies:
        weight_c = total_samples / (num_classes * count_c)

    Args:
        class_counts: Dictionary mapping model label (int) to its frequency count.
        device: The device to place the weight tensor on.
        label_smoothing: Optional label smoothing factor.

    Returns:
        nn.CrossEntropyLoss initialized with class weights.
    """
    if not class_counts:
        raise ValueError("class_counts cannot be empty.")

    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())

    sorted_classes = sorted(class_counts.keys())

    weights = []
    for cls in sorted_classes:
        count = class_counts[cls]
        if count == 0:
            raise ValueError(f"Class {cls} has count 0, cannot calculate inverse frequency.")
        weights.append(total_samples / (num_classes * count))

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    return nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
