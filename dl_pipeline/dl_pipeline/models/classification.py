import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
from efficientnet_pytorch import EfficientNet
import pretrainedmodels

class SEResNetx50_32(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super(SEResNetx50_32, self).__init__()
        self.encoder = pretrainedmodels.models.senet.se_resnext50_32x4d()

        self.encoder.dropout = nn.Dropout(dropout)
        self.encoder.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.encoder.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        
        return x

class SEResNetx101_32(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super(SEResNetx101_32, self).__init__()
        self.encoder = pretrainedmodels.models.senet.se_resnext101_32x4d()

        self.encoder.dropout = nn.Dropout(dropout)
        self.encoder.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.encoder.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        
        return x

class SENet154(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super(SENet154, self).__init__()
        self.encoder = pretrainedmodels.models.senet.senet154()

        self.encoder.dropout = nn.Dropout(dropout)
        self.encoder.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.encoder.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        
        return x

class PNasNetLarge(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super(PNasNetLarge, self).__init__()

        self.encoder = pretrainedmodels.models.pnasnet.pnasnet5large(num_classes=1000)
        self.encoder.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.encoder.last_linear = nn.Linear(4320, num_classes)

    def forward(self, x):
        x = self.encoder(x)

        return x
