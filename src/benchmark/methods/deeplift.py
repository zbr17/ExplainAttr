from typing import Tuple, Union, List
import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
from copy import deepcopy
from torchvision.models.vgg import VGG

import torch.nn as nn
from torch import Tensor
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from captum import attr

from .utils import norm_cam

class DeepLIFT:
    def __init__(self, model, device, **kwargs):
        self.device = device

        # initiate DeepLIFT by captum
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.dl = attr.DeepLift(model)
    
    def __call__(self, model: nn.Module, img: Tensor, layer_name: str) -> dict:
        # model preparation bs x C x H x W
        model.eval()

        # select class
        BS, C, H, W = img.size()
        output = model(img) # bs x K
        pred = torch.argmax(output, dim=-1)

        # run DeepLIFT by captum
        img.requires_grad_(True)
        cam = self.dl.attribute(img, target=pred)
        cam = torch.sum(cam, dim=1)

        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }