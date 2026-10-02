import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, vgg19, VGG19_Weights

try:
    from .base_model import BrainTumorModel
except ImportError:
    from base_model import BrainTumorModel


class ResNet18BrainTumor(BrainTumorModel):
    """
    ResNet18 adapted for 3-class brain tumor classification.
    Supports ImageNet pretrained initialization for end-to-end fine-tuning.
    """

    HEAD_PREFIXES = ("model.fc",)

    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VGG19BrainTumor(BrainTumorModel):
    """
    VGG19 adapted for 3-class brain tumor classification.
    Replaces the final 1000-way linear layer with `num_classes` outputs.
    """

    HEAD_PREFIXES = ("model.classifier.6",)

    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = VGG19_Weights.DEFAULT if pretrained else None
        self.model = vgg19(weights=weights)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
