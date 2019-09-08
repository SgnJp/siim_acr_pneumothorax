import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
from efficientnet_pytorch import EfficientNet
import pretrainedmodels

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Unet34( nn.Module ):
    def __init__(self):
        super(Unet34, self).__init__()
        self.base_model = models.resnet34(pretrained=True)
        self.up1 = up(768, 128)
        self.up2 = up(256, 128)
        self.up3 = up(192, 64)
        self.up4 = up(67, 32)
        self.last = nn.Conv2d(32, 1, 1)
        
    def forward(self, inputs):
        x1 = self.base_model.conv1(inputs)
        x1 = self.base_model.bn1(x1)
        x1 = self.base_model.relu(x1)

        x2 = self.base_model.layer1(x1)
        x3 = self.base_model.layer2(x2)
        x4 = self.base_model.layer3(x3)
        x5 = self.base_model.layer4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, inputs)
        x = self.last(x)
        
        return x


class EffUnetB1( nn.Module ):
    def __init__(self):
        super(EffUnet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')
        self.up1 = up(432, 40)
        self.up2 = up(80, 24)
        self.up3 = up(48, 16)
        self.up4 = up(32, 16)
        self.up5 = up(19, 16)
        self.last = nn.Conv2d(16, 1, 1)


    def forward(self, inputs):
        x0 = self.model._conv_stem(inputs)
        x0 = self.model._bn0(x0)

        xs = [x0]

        for block in self.model._blocks:
            xs.append(block(xs[-1]))

        x = self.up1(xs[23], xs[16])
        x = self.up2(x, xs[8])
        x = self.up3(x, xs[5])
        x = self.up4(x, xs[2])
        x = self.up5(x, inputs)
        x = self.last(x)
        
        return x

class EffUnetB2( nn.Module ):
    def __init__(self):
        super(EffUnetB2, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2')
        self.up1 = up(472, 48)
        self.up2 = up(96, 24)
        self.up3 = up(48, 16)
        self.up4 = up(32, 16)
        self.up5 = up(19, 16)
        self.last = nn.Conv2d(16, 1, 1)


    def forward(self, inputs):
        x0 = self.model._conv_stem(inputs)
        x0 = self.model._bn0(x0)

        xs = [x0]

        for block in self.model._blocks:
            xs.append(block(xs[-1]))

        x = self.up1(xs[23], xs[16])
        x = self.up2(x, xs[8])
        x = self.up3(x, xs[5])
        x = self.up4(x, xs[2])
        x = self.up5(x, inputs)
        x = self.last(x)
        
        return x

class EffUnetB3( nn.Module ):
    def __init__(self):
        super(EffUnetB3, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.up1 = up(520, 48)
        self.up2 = up(96, 32)
        self.up3 = up(64, 24)
        self.up4 = up(48, 16)
        self.up5 = up(19, 16)
        self.last = nn.Conv2d(16, 1, 1)


    def forward(self, inputs):
        x0 = self.model._conv_stem(inputs)
        x0 = self.model._bn0(x0)

        xs = [x0]

        for block in self.model._blocks:
            xs.append(block(xs[-1]))

        x = self.up1(xs[26], xs[18])
        x = self.up2(x, xs[8])
        x = self.up3(x, xs[5])
        x = self.up4(x, xs[2])
        x = self.up5(x, inputs)
        x = self.last(x)
        
        return x


class EffUnetB4( nn.Module ):
    def __init__(self):
        super(EffUnetB4, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.up1 = up(608, 56)
        self.up2 = up(112, 32)
        self.up3 = up(64, 24)
        self.up4 = up(48, 16)
        self.up5 = up(19, 16)
        self.last = nn.Sequential(
            nn.Conv2d(16, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1))


    def forward(self, inputs):
        x0 = self.model._conv_stem(inputs)
        x0 = self.model._bn0(x0)

        xs = [x0]

        for block in self.model._blocks:
            xs.append(block(xs[-1]))

        x = self.up1(xs[32], xs[22])
        x = self.up2(x, xs[10])
        x = self.up3(x, xs[6])
        x = self.up4(x, xs[2])
        x = self.up5(x, inputs)
        x = self.last(x)
        
        return x

class EffUnetB5( nn.Module ):
    def __init__(self):
        super(EffUnetB5, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.up1 = up(688, 176)
        self.up2 = up(240, 64)
        self.up3 = up(104, 48)
        self.up4 = up(72, 24)
        self.up5 = up(27, 16)
        self.last = nn.Sequential(
            nn.Conv2d(16, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1))


    def forward(self, inputs):
        x0 = self.model._conv_stem(inputs)
        x0 = self.model._bn0(x0)

        xs = [x0]

        for block in self.model._blocks:
            xs.append(block(xs[-1]))

        x = self.up1(xs[39], xs[27])
        x = self.up2(x, xs[13])
        x = self.up3(x, xs[8])
        x = self.up4(x, xs[3])
        x = self.up5(x, inputs)
        x = self.last(x)
        
        return x