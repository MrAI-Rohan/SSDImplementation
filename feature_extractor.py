import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import v2
from collections import OrderedDict


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, **kwargs)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.relu(self.conv(x))


class SSDFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in
                                                enumerate(self.model.features) 
                                                if isinstance(layer, nn.MaxPool2d))

        self.model.features[maxpool3_pos].ceil_mode = True

        # Layers till conv4_3
        self.conv4_3 = self.model.features[:maxpool4_pos]

        # Layers till conv7
        self.conv7 = nn.Sequential(
                        *self.model.features[maxpool4_pos:-1],
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        ConvBlock(in_channels=512, out_channels=1024, kernel_size=3,
                                    padding=6, dilation=6),
                        ConvBlock(in_channels=1024, out_channels=1024, kernel_size=1),
                    )


        self.extra = nn.ModuleList([
                    nn.Sequential(
                        OrderedDict([
                        ('conv8_1', ConvBlock(in_channels=1024, out_channels=256, kernel_size=1)),
                        ('conv8_2', ConvBlock(in_channels=256, out_channels=512, kernel_size=3,
                                                stride=2, padding=1))
                    ])),

                    nn.Sequential(
                        OrderedDict([
                        ('conv9_1', ConvBlock(in_channels=512, out_channels=128, kernel_size=1)),
                        ('conv9_2', ConvBlock(in_channels=128, out_channels=256, kernel_size=3,
                                                stride=2, padding=1))
                    ])),

                    nn.Sequential(
                        OrderedDict([
                        ('conv10_1', ConvBlock(in_channels=256, out_channels=128, kernel_size=1)),
                        ('conv10_2', ConvBlock(in_channels=128, out_channels=256, kernel_size=3))
                    ])),

                    nn.Sequential(
                        OrderedDict([
                        ('conv11_1', ConvBlock(in_channels=256, out_channels=128, kernel_size=1)),
                        ('conv11_2', ConvBlock(in_channels=128, out_channels=256, kernel_size=3))
                    ]))
                ])


    def forward(self, x):
        outputs = []
        x = self.conv4_3(x)

        # L2 normaliztion with scale weight
        outputs.append(F.normalize(x) * self.scale_weight)

        x = self.conv7(x)
        outputs.append(x)

        for layer in self.extra:
            x = layer(x)
            outputs.append(x)

        layer_names = ["conv4_3", "conv7"] + [f"conv{i+8}_2" for i in range(len(self.extra))]

        return OrderedDict(zip(layer_names, outputs))
