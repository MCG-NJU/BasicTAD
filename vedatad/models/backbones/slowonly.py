from torch.nn import ModuleList
import torch
import torch.nn as nn
import os
import math
from collections import OrderedDict
from vedacore.misc import registry

@registry.register_module('backbone')
class SlowOnly(nn.Module):
    
    def __init__(self, num_layers=50,freeze_bn=True, freeze_bn_affine=True,kernel_size=(1,1,1),stride=(1,1,1)):
        super(SlowOnly, self).__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        model_name = 'slow_r'+str(num_layers)
        model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
        self._modules = OrderedDict()
        self._modules['blocks'] = ModuleList()
        for i in range(5):
            self._modules['blocks'].append(model._modules['blocks'][i])
        self._model=model
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine
        self._modules['blocks'].append(nn.MaxPool3d(self.kernel_size, self.stride))

    def forward(self, x):
        for block in self._modules['blocks']:
            x = block(x)
        return x

    def train(self, mode=True):
        super(SlowOnly, self).train(mode)
        if self._freeze_bn and mode:
            for _, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)