import torch.nn as nn
import torch.nn.functional as F


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_bits_list):
        super(USBatchNorm2d, self).__init__(num_features)
        
        self.num_features = num_features
        self.num_bits_list = num_bits_list

        self.num_bits = None

        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(num_features) for _ in self.num_bits_list]
        )


    def set_precision(self, num_bits=None):
        self.num_bits = num_bits


    def forward(self, input):
        weight = self.weight
        bias = self.bias

        if self.num_bits == 0:
            idx = -1
        else:
            assert self.num_bits in self.num_bits_list
            idx = self.num_bits_list.index(self.num_bits)

        y = self.bn[idx](input)
        
        return y
