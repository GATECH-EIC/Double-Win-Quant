import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np

from .quantize import QConv2d, QLinear
from .slimmable_ops import USBatchNorm2d

BatchNorm2d = nn.BatchNorm2d

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.conv1 = QConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = QConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x, full_prec=True) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.conv1 = QConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = BatchNorm2d(planes)
        self.conv3 = QConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = QConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)


    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x, full_prec=True) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, normalize=None):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = QConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.normalize = normalize

        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
            
        out = self.conv1(x, full_prec=True)

        for op in self.layer1:
            out = op(out)

        for op in self.layer2:
            out = op(out)

        for op in self.layer3:
            out = op(out)

        for op in self.layer4:
            out = op(out)

        out = F.relu(self.bn(out))
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


def PreActResNet18(num_bits_list=None, num_classes=10, normalize=None):
    global BatchNorm2d
    if num_bits_list is not None:
        BatchNorm2d = lambda num_features: USBatchNorm2d(num_features, num_bits_list)

    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, normalize=normalize)
