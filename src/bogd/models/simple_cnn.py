from typing import List

import torch.nn as nn
import torch
from torch import Tensor

# TODO: to be compatible with LRP 
from torch.nn import Conv2d as ConvLayer, ReLU as GuidedReLULayer, LeakyReLU as LeakyReLULayer, Linear as LinearLayer
# from .layers import ConvLayer, GuidedReLULayer, LeakyReLULayer, LinearLayer
from .tracable_module import TracableModule

def give_act(act_type="leaky"):
    if act_type == "leaky":
        return LeakyReLULayer(inplace=True)
    elif act_type == "guided":
        return GuidedReLULayer(inplace=True)
    else:
        raise KeyError(f"Invalid act type: {act_type}")

class SimpleCNN(TracableModule):
    def __init__(
        self,
        in_dim: int = 1,
        in_size: int = 28,
        out: List[int] = None,
        act_type: str = "guided",
    ):
        """
        Args:
            in_dim (int): the input dimension (defualt: 1(MNIST))
            in_size (int): the size of input data (default: 28(MNIST))
            out (list): recommended setting: [8, 32, 512, 2048], which is an experimental tradeoff choice for BOGD optimizer.
        """
        super(SimpleCNN, self).__init__()
        self.in_dim = in_dim
        self.in_size = in_size
        self.act_type = act_type
        if out is None:
            if in_dim == 3:
                out = [24, 64, 512, 1536, 4608]
            elif in_dim == 1:
                out = [8, 24, 288, 864, 2592]
            # out = [8, 24, 288, 864, 2592]

        self.conv1 = ConvLayer(in_dim, out[0], 5, padding=1)
        self.act1 = give_act(act_type)
        self.apool1 = ConvLayer(out[0], out[1], 2, 2)

        self.conv2 = ConvLayer(out[1], out[2], 4, padding=1)
        self.act2 = give_act(act_type)
        self.apool2 = ConvLayer(out[2], out[3], 2, 2)

        self.conv3 = ConvLayer(out[3], out[4], 3, padding=1)
        self.act3 = give_act(act_type)

        # self.gpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc = LinearLayer(out[4], 10)

    def forward(self, x):
        # input: 1 x 4k x 4k
        x = self.act1(self.conv1(x)) # c x (4k-2) x (4k-2)
        x = self.apool1(x) # c x (2k-1) x (2k-1)
        x = self.act2(self.conv2(x)) # c x (2k-2) x (2k-2)
        x = self.apool2(x) # c x (k-1) x (k-1)
        x = self.act3(self.conv3(x)) # c x (k-1) x (k-1)
        x = self.mpool(x) # + self.gpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    # @torch.no_grad()
    # def trace_back(
    #     self, 
    #     x, 
    #     module_name: str, 
    #     base_x: Tensor = None,
    #     infeat_dict: dict = None,
    #     return_all: bool = False
    # ):
    #     """
    #     Args:
    #         x (Tensor): data to trace back.
    #         module_name (str): the module name at the top layer to trace back
    #         base_x (Tensor): baseline.
    #         infeat_dict (dict): dictionary of input features at different layers
    #         return_all (bool): whether to return the intermediate variables
    #     """
    #     def _traceback(x, module: nn.Module):
    #         assert isinstance(module, TracableModule)
    #         return module.trace_back(x)
        
    #     recon_dict = {}
    #     module_list = []
    #     for name, _ in self.named_children():
    #         if name != module_name:
    #             module_list.append(name)
    #         else:
    #             break
        
    #     for _ in range(len(module_list)):
    #         top_name = module_list.pop(-1)
    #         sub_module = getattr(self, top_name)
    #         x = _traceback(x, sub_module)
    #         # NOTE: New method - data fusion
    #         if (infeat_dict is not None) and (base_x is not None):
    #             # NOTE: mimics Kalman filtering algorithm
    #             # ori_x = infeat_dict[top_name][0]
    #             # e_prd = (ori_x - x).pow(2)
    #             # e_mea = ori_x.pow(2)
    #             # e_prd = e_prd / (e_prd + e_mea + 1e-8)
    #             # x = e_prd * x + (1 - e_prd) * ori_x
    #             # NOTE: traceback with rectification
    #             ori_in_feat = infeat_dict[top_name][0]
    #             base_x = _traceback(base_x, sub_module)
    #             x = x - base_x + ori_in_feat
    #             base_x = ori_in_feat
    #         recon_dict[top_name] = x
        
    #     if return_all:
    #         return x, recon_dict
    #     else:
    #         return x

if __name__ == "__main__":
    model = SimpleCNN(1, 28)
    data = torch.randn(10, 1, 28, 28)
    output = model(data) 
