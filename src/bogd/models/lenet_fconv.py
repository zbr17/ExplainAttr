"""
Modified from: https://github.com/pskugit/custom-conv2d/blob/master/models/customconv.py
"""
import torch.nn as nn
import torch

from .layers import ConvLayer, ReLULayer, MaxPoolLayer
from .tracable_module import TracableModule

class LeNetFullConv(TracableModule):
    def __init__(
        self,
        in_dim: int = 1,
        in_size: int = 28
    ):
        super(LeNetFullConv, self).__init__()
        self.in_dim = in_dim
        self.in_size = in_size
        self.conv1 = ConvLayer(in_dim, 8, 5)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolLayer(2)
        self.conv2 = ConvLayer(8, 32, 5)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolLayer(2)
        self.conv3 = ConvLayer(32, 128, 3, padding=1)
        self.relu3 = ReLULayer()
        self.conv4 = ConvLayer(128, 64, 3, padding=1)
        self.relu4 = ReLULayer()
        self.conv5 = ConvLayer(64, 10, 3, padding=1)
        self.gpool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.gpool(x)
        x = x.reshape(x.size(0), -1)
        return x

    @torch.no_grad()
    def trace_back(self, x, module_name: str):
        self._entry_point = False
        def _traceback(x, module: nn.Module):
            if (not self._entry_point) and id(module) == id(getattr(self, module_name)):
                self._entry_point = True
            
            if self._entry_point:
                assert isinstance(module, TracableModule)
                return module.trace_back(x)
            else:
                return x
        
        x = _traceback(x, self.gpool)
        x = _traceback(x, self.conv5)
        x = _traceback(x, self.relu4)
        x = _traceback(x, self.conv4)
        x = _traceback(x, self.relu3)
        x = _traceback(x, self.conv3)
        x = _traceback(x, self.pool2)
        x = _traceback(x, self.relu2)
        x = _traceback(x, self.conv2)
        x = _traceback(x, self.pool1)
        x = _traceback(x, self.relu1)
        x = _traceback(x, self.conv1)

        return x


