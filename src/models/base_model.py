import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BrainTumorModel(nn.Module, ABC):
    """
    Abstract base class for all Brain Tumor CNN Classification models.
    Enforces a consistent interface across different architectures.
    """

    # Parameter-name prefixes that belong to the classification head.
    # Backbone parameters (everything not matching one of these prefixes)
    # are frozen during the initial training phase and unfrozen for
    # fine-tuning. Subclasses override this.
    HEAD_PREFIXES: tuple[str, ...] = ()

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

    def is_head_parameter(self, name: str) -> bool:
        """Whether a named parameter belongs to the classification head."""
        return any(name.startswith(prefix) for prefix in self.HEAD_PREFIXES)

    def head_parameter_names(self) -> list[str]:
        return [name for name, _ in self.named_parameters() if self.is_head_parameter(name)]

    def head_parameters(self):
        return [param for name, param in self.named_parameters() if self.is_head_parameter(name)]

    def freeze_backbone(self) -> None:
        """Freeze everything except the classification head."""
        for name, param in self.named_parameters():
            param.requires_grad = self.is_head_parameter(name)

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
            "trainable_parameters": self.count_parameters(),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
