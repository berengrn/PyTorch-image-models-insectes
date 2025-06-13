import torch
import torch.nn as nn
import torch.nn.functional as F

from ._registry import register_model

class taxaNetModel(nn.Module):

    def __init__(self):
        from timm.models import create_model
        super().__init__()
        self.num_classes = 4
        self.backbone = create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,4),
        )

    def forward(self,x):
        x = self.backbone(x)
        return self.classifier(x)

@register_model
def taxaNet(**kwargs):
    model = taxaNetModel()
    return model

        

