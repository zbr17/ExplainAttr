import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from ..tracable_module import TracableModule

class LeakyReLULayer(TracableModule):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, slope: float = 0.5, inplace: bool = False):
        super(LeakyReLULayer, self).__init__()
        self.slope = slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.slope, self.inplace)
    
    @torch.no_grad()
    def trace_back(self, input: Tensor) -> Tensor:
        zeros = torch.zeros_like(input)
        output = torch.maximum(input, zeros) + torch.minimum(input, zeros) / self.slope
        return output