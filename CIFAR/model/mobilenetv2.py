import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np

from .quantize import QConv2d, QLinear
from .slimmable_ops import USBatchNorm2d

BatchNorm2d = nn.BatchNorm2d


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = QConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = QConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                QConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2_Net(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, normalize=None):
        super(MobileNetV2_Net, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = QConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = QConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        self.normalize = normalize

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def set_precision(self, num_bits=None, num_grad_bits=None):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.set_precision(num_bits, num_grad_bits)
            if isinstance(module, USBatchNorm2d):
                module.set_precision(num_bits)


def MobileNetV2(num_bits_list=None, num_classes=10, normalize=None):
    global BatchNorm2d
    if num_bits_list is not None:
        BatchNorm2d = lambda num_features: USBatchNorm2d(num_features, num_bits_list)

    return MobileNetV2_Net(num_classes=num_classes, normalize=normalize)