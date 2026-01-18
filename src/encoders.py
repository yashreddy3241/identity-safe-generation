import torch
import torch.nn as nn
import torchvision.models as models

class FactorEncoder(nn.Module):
    """
    Predicts a specific factor vector from an input image.
    Uses a lightweight ResNet18 backbone.
    """
    def __init__(self, output_dim=64):
        super().__init__()
        # Use a small backbone
        self.backbone = models.resnet18(pretrained=True)
        # Replace fc layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, output_dim)
        
    def forward(self, x):
        return self.backbone(x)

class MultiFactorEncoder(nn.Module):
    """
    Wrapper that contains one encoder per factor.
    """
    def __init__(self, factor_dims):
        # factor_dims: dict mapping 'geometry' -> int dim, etc.
        super().__init__()
        self.encoders = nn.ModuleDict()
        for name, dim in factor_dims.items():
            self.encoders[name] = FactorEncoder(output_dim=dim)
            
    def forward(self, x):
        outputs = {}
        for name, encoder in self.encoders.items():
            outputs[name] = encoder(x)
        return outputs
