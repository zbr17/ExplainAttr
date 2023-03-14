import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast

from .layers import ConvLayer, LeakyReLULayer, LinearLayer
from .tracable_module import TracableModule

__all__ = ['vgg16']

class VGG(TracableModule):
    """
    Modified from the original VGG Net. There are two changes:
    (1) Replace the Maxpooling-layer with Convolutional-layer (2 stride + 2 kernel-size).
    (2) Replace the ReLU activation with LeakyReLU activation for reversible operation.
    """
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            LinearLayer(512 * 7 * 7, 4096),
            LeakyReLULayer(inplace=True),
            nn.Dropout(),
            LinearLayer(4096, 4096),
            LeakyReLULayer(inplace=True),
            nn.Dropout(),
            LinearLayer(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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

    @torch.no_grad()
    def trace_back(self, x, module_name: int, return_all: bool = False):
        """
        Only support the trace back of 'features' sub-module
        """
        recon_dict = {}
        module_name = int(module_name)
        assert module_name <= len(self.features)
        module_list = list(range(module_name-1, -1, -1))
        for idx in module_list:
            sub_module = self.features[idx]
            assert isinstance(sub_module, TracableModule)
            x = sub_module.trace_back(x)
            recon_dict[str(idx)] = x
        
        if return_all:
            return x, recon_dict
        else:
            return x

        


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers += [ConvLayer(in_channels, in_channels, kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = ConvLayer(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), LeakyReLULayer(inplace=True)]
            else:
                layers += [conv2d, LeakyReLULayer(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, progress: bool, **kwargs: Any) -> VGG:
    kwargs['init_weights'] = True
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg16(progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, progress, **kwargs)
