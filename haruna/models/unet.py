from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class OrderedModuleDict(nn.ModuleDict):

    def __init__(self, *args, **kwargs):
        super(OrderedModuleDict, self).__init__()
        self.update(modules=OrderedDict(*args, **kwargs))


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.operations = OrderedModuleDict([
            # down 1
            ('conv1_1', ConvBNReLU(3, 64, 3)),
            ('conv1_2', ConvBNReLU(64, 64, 3)),
            ('pool1', nn.MaxPool2d(2, 2)),
            # down 2
            ('conv2_1', ConvBNReLU(64, 128, 3)),
            ('conv2_2', ConvBNReLU(128, 128, 3)),
            ('pool2', nn.MaxPool2d(2, 2)),
            # down 3
            ('conv3_1', ConvBNReLU(128, 256, 3)),
            ('conv3_2', ConvBNReLU(256, 256, 3)),
            ('pool3', nn.MaxPool2d(2, 2)),
            # down 4
            ('conv4_1', ConvBNReLU(256, 512, 3)),
            ('conv4_2', ConvBNReLU(512, 512, 3)),
            ('pool4', nn.MaxPool2d(2, 2)),
            # bottleneck 5
            ('conv5_1', ConvBNReLU(512, 1024, 3)),
            ('conv5_2', ConvBNReLU(1024, 1024, 3)),
            # up 6
            ('deconv6', nn.ConvTranspose2d(1024, 512, 2, stride=2)),
            ('conv6_1', ConvBNReLU(1024, 512, 3)),
            ('conv6_2', ConvBNReLU(512, 512, 3)),
            # up 7
            ('deconv7', nn.ConvTranspose2d(512, 256, 2, stride=2)),
            ('conv7_1', ConvBNReLU(512, 256, 3)),
            ('conv7_2', ConvBNReLU(256, 256, 3)),
            # up 8
            ('deconv8', nn.ConvTranspose2d(256, 128, 2, stride=2)),
            ('conv8_1', ConvBNReLU(256, 128, 3)),
            ('conv8_2', ConvBNReLU(128, 128, 3)),
            # up 9
            ('deconv9', nn.ConvTranspose2d(128, 64, 2, stride=2)),
            ('conv9_1', ConvBNReLU(128, 64, 3)),
            ('conv9_2', ConvBNReLU(64, 64, 3)),
            # out 10
            ('conv10', nn.Conv2d(64, 3, 1))
        ])

    def forward(self, x):
        features = []
        for k, v in self.operations.items():
            if k.startswith('pool'):
                features.append(x)

            x = v(x)

            if k.startswith('deconv'):
                skip = center_crop(features.pop(), x)
                x = torch.cat([skip, x], dim=1)
        return x


def center_crop(input, target):
    d2 = target.size(2) - input.size(2)
    d3 = target.size(3) - input.size(3)
    pad = (d3 // 2, d3 - d3 // 2, d2 // 2, d2 - d2 // 2)
    return F.pad(input, pad=pad)
