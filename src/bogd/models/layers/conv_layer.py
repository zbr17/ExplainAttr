"""
Modified from: https://github.com/pskugit/custom-conv2d/blob/master/models/customconv.py
"""
import torch.nn as nn
import torch
from typing import Optional, Tuple, Union
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch import Tensor
import torch.nn.functional as F

import tbwriter

from ..tracable_module import TracableModule

_size_2_t = Union[int, Tuple[int, int]]

class ConvLayer(_ConvNd, TracableModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros"
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ConvLayer, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

    def _conv2d_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        batch_size, in_channels, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size
        out_h = int((in_h - k_h + 2 * self.padding[0]) / self.stride[0] + 1)
        out_w = int((in_w - k_w + 2 * self.padding[1]) / self.stride[1] + 1)
        assert in_channels == self.in_channels
        input_unfold = self.unfold(input) # bs x (in x kh x kw) x (oh x ow)
        _weight = weight.view(weight.size(0), -1).t() # out x in x kh x kw -> (in x kh x kw) x out
        tbwriter.update_conv_norm(self, _weight, "weight.norm")
        tbwriter.update_conv_val_norm(self, input_unfold, "val.norm")
        if bias is None:
            out_unfold = input_unfold.transpose(1, 2).matmul(_weight).transpose(1, 2)
            tbwriter.update_conv_vars(self, out_unfold, "no-bias")
        else:
            out_unfold = input_unfold.transpose(1, 2).matmul(_weight) # bs x (oh x ow) x out
            tbwriter.update_conv_vars(self, out_unfold, "before-bias")
            out_unfold = out_unfold + bias # bs x (oh x ow) x out
            tbwriter.update_conv_vars(self, out_unfold, "after-bias")
            out_unfold = out_unfold.transpose(1, 2)
        out = out_unfold.view(batch_size, self.out_channels, out_h, out_w) # bs x out x (oh x ow) -> bs x out x oh x ow

        return out
    
    def forward(self, input: Tensor) -> Tensor:
        return self._conv2d_forward(input, self.weight, self.bias)
    
    @torch.no_grad()
    def trace_back(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): size = bs x out x oh x ow
        """
        batch_size, in_channels, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size
        out_h = int(self.stride[0] * (in_h - 1) - 2 * self.padding[0] + k_h)
        out_w = int(self.stride[1] * (in_w - 1) - 2 * self.padding[1] + k_w)
        
        input_unfold = input.view(batch_size, in_channels, -1) # bs x out x oh x ow -> bs x out x (oh x ow)

        _weight = self.weight.view(self.weight.size(0), -1).t() # out x in x kh x kw -> (in x kh x kw) x out

        if self.bias is None:
            input_unfold = input_unfold.transpose(1, 2) # bs x (oh x ow) x out
            _weight_norm = _weight.norm(p=2, dim=0, keepdim=True).pow(2) # 1 x out
            input_unfold = input_unfold / (_weight_norm + 1e-8) # bs x (oh x ow) x out
            input_unfold = input_unfold.matmul(_weight.t()).transpose(1, 2) # bs x (oh x ow) x out -> bs x (in x kh x kw) x (oh x ow)
        else:
            input_unfold = input_unfold.transpose(1, 2) # bs x (oh x ow) x out
            input_unfold = input_unfold - self.bias # bs x (oh x ow) x out
            _weight_norm = _weight.norm(p=2, dim=0, keepdim=True).pow(2) # 1 x out
            input_unfold = input_unfold / (_weight_norm + 1e-8) # bs x (oh x ow) x out
            input_unfold = input_unfold.matmul(_weight.t()).transpose(1, 2) # bs x (oh x ow) x out -> bs x (in x kh x kw) x (oh x ow)
        
        fold = nn.Fold(output_size=(out_h, out_w), 
                    kernel_size=self.kernel_size, dilation=self.dilation,
                    padding=self.padding, stride=self.stride)
        input = fold(input_unfold)
        # bs x (in x kh x kw) x (oh x ow) -> bs x in x ih x iw
        divisor = fold(self.unfold(torch.ones_like(input)))
        input = input / divisor
        
        return input