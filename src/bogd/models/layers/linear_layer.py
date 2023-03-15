import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
import torch.nn.functional as F

from ..tracable_module import TracableModule

class LinearLayer(TracableModule):
    def __init__(
        self,
        in_features: int, 
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
    
    @torch.no_grad()
    def trace_back(self, input: Tensor) -> Tensor:
        if self.bias is not None:
            output = input - self.bias
        output = output @ self.weight
        return output
        