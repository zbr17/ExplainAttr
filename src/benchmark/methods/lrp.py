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

from . import lrp_utils
from captum import attr

from .utils import norm_cam

class LRP:
    def __init__(self, model, device, **kwargs):
        self.device = device

        # initiate LRP by TorchLRP
        if isinstance(model, VGG):
            self.lrp_model = lrp_utils.convert_vgg(model).to(self.device)
        else:
            self.lrp_model = lrp_utils.convert_resnet(model).to(self.device)

        # # initiate LRP by captum
        # for module in model.modules():
        #     if isinstance(module, nn.ReLU):
        #         module.inplace = False
        # self.lrp = attr.LRP(model)

    def __call__(self, model: nn.Module, img: Tensor, layer_name: str) -> dict:
        # model preparation bs x C x H x W
        model.eval()

        # select class
        BS, C, H, W = img.size()
        output = model(img) # bs x K
        pred = torch.argmax(output, dim=-1)

        # run LRP by TorchLRP
        img.requires_grad_(True)
        output = self.lrp_model.forward(img, explain=True, rule="epsilon")
        scores = output[torch.arange(BS), pred].sum()
        scores.backward()
        cam = torch.sum(img.grad, dim=1)

        # # run LRP by captum
        # img.requires_grad_(True)
        # cam = self.lrp.attribute(img, target=pred)
        # cam = torch.sum(cam, dim=1)

        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }