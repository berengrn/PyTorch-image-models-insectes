import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model,registry

class taxaNet(nn.Module):

    def __init__(self):
        super.__init__()
        self.backbone = create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1280,512),
            nn.ReLU(),
            nn.Droupout(p=0.5),
            nn.Linear(512,4),
        )

    def _forward(self,x):
        return self.backbnone(x)

@register_model
def taxaNet(**kwargs):
    model = taxaNet()
    return model

        

