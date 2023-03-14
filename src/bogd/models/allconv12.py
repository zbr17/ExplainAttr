from typing import Optional

import torch
import torch.nn as nn

from .layers import ConvLayer, GuidedReLULayer, LeakyReLULayer, LinearLayer
from .tracable_module import TracableModule

def give_act(act_type="leaky"):
    if act_type == "leaky":
        return LeakyReLULayer(inplace=True)
    elif act_type == "guided":
        return GuidedReLULayer(inplace=True)
    else:
        raise KeyError(f"Invalid act type: {act_type}")

class AllConv12(TracableModule):
    """
    This is the implementation of `Striving for Simplicity: All Convolutional Networks` according to `Table 6: Architecture of the ImageNet network`.
    """
    def __init__(
        self,
        num_classes: int = 1000,
        setting: Optional[str] = None,
        init_weights: bool = True,
        act_type: str = "guided",
    ):
        super(AllConv12, self).__init__()
        self.num_classes = num_classes
        self.act_type = act_type
        
        padding_list = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        stride_list = [3, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1]
        kernel_list = [10, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 1]

        if setting == "base":
            channel_list = [3, 96, 96, 96, 256, 256, 256, 384, 384, 384, 1024, 1024, num_classes]
        elif setting == "large":
            channel_list = [3, 100, 50, 90, 162, 81, 243, 729, 243, 729, 2187, 1458, num_classes]
        else:
            raise KeyError(f"Invalid setting: {setting}")
        
        for idx in range(12):
            setattr(self, f"conv{idx+1}", 
                ConvLayer(channel_list[idx], channel_list[idx+1], 
                kernel_size=kernel_list[idx], stride=stride_list[idx], 
                padding=padding_list[idx])
            )
            setattr(self, f"act{idx+1}", give_act(act_type))

        self.dropout = nn.Dropout(p=0.5)
        self.apool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, ConvLayer)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, LinearLayer)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.act7(self.conv7(x))
        x = self.act8(self.conv8(x))
        x = self.act9(self.conv9(x))
        x = self.dropout(x)
        x = self.act10(self.conv10(x))
        x = self.act11(self.conv11(x))
        x = self.act12(self.conv12(x))
        x = self.apool(x)
        x = x.view(x.size(0), -1)
        return x
    
    @property
    def exclude_layer(self):
        return ["dropout", "apool"]
    
    @torch.no_grad()
    def trace_back(self, x, module_name: str, return_all: bool = False):
        """
        Except for "dropout" and "apool" layers.
        """
        recon_dict = {}
        module_list = []
        for name, _ in self.named_children():
            if name != module_name:
                if name not in self.exclude_layer:
                    module_list.append(name)
            else:
                break
        
        for _ in range(len(module_list)):
            top_name = module_list.pop(-1)
            sub_module = getattr(self, top_name)
            x = sub_module.trace_back(x)
            recon_dict[top_name] = x
        
        if return_all:
            return x, recon_dict
        else:
            return x