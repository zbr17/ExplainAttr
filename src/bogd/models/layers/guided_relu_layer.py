import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from ..tracable_module import TracableModule

class GuidedReLULayer(TracableModule):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(GuidedReLULayer, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if self.tracing:
            self.forward_mask = input > 0
        return F.relu(input, inplace=self.inplace)
    
    @torch.no_grad()
    def trace_back(self, input: Tensor) -> Tensor:
        assert self.tracing
        # compute mask
        backward_mask = input > 0
        mask = (self.forward_mask & backward_mask).float()
        input = mask * input
        # self.forward_mask = None # clear the previous forward positions
        return input