import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BrainTumorModel(nn.Module, ABC):
    """
    Abstract base class for all Brain Tumor CNN Classification models.
    Enforces a consistent interface across different architectures.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        pass

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_all_layers(self) -> None:
        """Freezes all layers in the model (useful for feature extraction)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self) -> None:
        """Unfreezes all layers in the model."""
        for param in self.parameters():
            param.requires_grad = True

    def summary(self) -> dict:
        """Returns a brief summary of the model."""
        return {
            "name": self.__class__.__name__,
            "trainable_parameters": self.count_parameters()
        }
