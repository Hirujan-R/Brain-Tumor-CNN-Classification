import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

try:
    from .base_model import BrainTumorModel
except ImportError:
    from base_model import BrainTumorModel

class GoogLeNetBrainTumor(BrainTumorModel):
    """
    GoogLeNet (Inception v1) reproduction for Brain Tumor Classification.
    By default, instantiates a fresh model without pre-trained weights to
    fully reproduce training from scratch as requested, but allows using
    pre-trained weights if desired.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = False, aux_logits: bool = True):
        super().__init__()
        
        # Load the base GoogLeNet model
        weights = GoogLeNet_Weights.DEFAULT if pretrained else None
        
        # Note: If weights are used, torchvision forces aux_logits=True for GoogLeNet if we want to
        # retain the exact pretrained auxiliary heads, but we can override and ignore them later.
        # For training from scratch, we typically want aux_logits to help with vanishing gradients.
        # But if the user specifies False, we respect that.
        self.model = googlenet(weights=weights, aux_logits=aux_logits, transform_input=True)
        
        # The original output is 1000 classes (ImageNet). We need to change the final fully connected layers.
        
        # Replace main classifier head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # Replace auxiliary classifier heads if they exist
        if aux_logits and self.model.aux1 is not None:
            self.model.aux1.fc2 = nn.Linear(self.model.aux1.fc2.in_features, num_classes)
        if aux_logits and self.model.aux2 is not None:
            self.model.aux2.fc2 = nn.Linear(self.model.aux2.fc2.in_features, num_classes)

        self.aux_logits = aux_logits

    def forward(self, x: torch.Tensor):
        """
        Forward pass. 
        During training with aux_logits=True, this returns a namedtuple (logits, aux_logits2, aux_logits1).
        During evaluation (model.eval()), this only returns the main logits.
        """
        return self.model(x)
