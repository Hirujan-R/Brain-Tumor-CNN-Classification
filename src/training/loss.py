import torch
import torch.nn as nn
from typing import Dict

def get_weighted_loss(
    class_counts: Dict[int, int],
    device: torch.device = torch.device('cpu')
) -> nn.CrossEntropyLoss:
    """
    Creates a Weighted CrossEntropyLoss to handle class imbalance.
    Weights are calculated inversely proportional to class frequencies:
        weight_c = total_samples / (num_classes * count_c)
        
    Args:
        class_counts: Dictionary mapping model label (int) to its frequency count.
                      e.g., {0: 1426, 1: 708, 2: 930}
        device: The device to place the weight tensor on.
        
    Returns:
        nn.CrossEntropyLoss initialized with class weights.
    """
    if not class_counts:
        raise ValueError("class_counts cannot be empty.")
        
    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())
    
    # Sort keys to ensure tensor aligns with class indices (0, 1, 2...)
    sorted_classes = sorted(class_counts.keys())
    
    # Calculate weights: weight_c = total_samples / (num_classes * count_c)
    weights = []
    for cls in sorted_classes:
        count = class_counts[cls]
        if count == 0:
            raise ValueError(f"Class {cls} has count 0, cannot calculate inverse frequency.")
        w = total_samples / (num_classes * count)
        weights.append(w)
        
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    
    return nn.CrossEntropyLoss(weight=weights_tensor)
