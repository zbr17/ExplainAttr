import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from ..tracable_module import TracableModule

class ReLULayer(TracableModule):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLULayer, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)
    
    @torch.no_grad()
    def trace_back(self, input: Tensor) -> Tensor:
        return input