from typing import Union, Tuple, Optional

import torch
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.nn.modules.pooling import _MaxPoolNd
import torch.nn.functional as F

from ..tracable_module import TracableModule

_size_2_t = Union[int, Tuple[int, int]]

class MaxPoolLayer(_MaxPoolNd, TracableModule):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']
    return_indices: bool
    ceil_mode: bool

    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPoolLayer, self).__init__(kernel_size, stride, padding, dilation, 
                                            return_indices, ceil_mode)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if (stride is not None) else _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        if not self.tracing:
            return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        else:
            output, indices = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
            self.trace_indices = indices
            return output
    
    def trace_setting(self):
        if self.tracing:
            self.return_indices = True
        else:
            self.return_indices = False
    
    # @torch.no_grad()
    # def trace_back(self, input: Tensor) -> Tensor:
    #     return F.max_unpool2d(input, self.trace_indices, self.kernel_size,
    #                         self.stride, self.padding)

    @torch.no_grad()
    def trace_back(self, input: Tensor) -> Tensor:
        # try upsample
        _, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size
        out_h = int(self.stride[0] * (in_h - 1) - 2 * self.padding[0] + k_h)
        out_w = int(self.stride[1] * (in_w - 1) - 2 * self.padding[1] + k_w)
        return F.upsample(input, size=(out_h, out_w), mode="nearest")
