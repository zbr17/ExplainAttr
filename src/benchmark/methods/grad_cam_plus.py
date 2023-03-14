from typing import Tuple, Union
import numpy as np
from scipy.ndimage.interpolation import zoom

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from .utils import norm_cam

class GradCAMPlus:
    """
    Refer to https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py
    """
    def __init__(self, **kwargs):
        pass
    
    def _register_model(self, model: nn.Module, layer_name: str):
        self.data_hidden = []
        # register the penultimate layer
        def hook(module, input, output):
            self.data_hidden.append(output)
        layer: nn.Module = eval(f"model.{layer_name}")
        self.handler = layer.register_forward_hook(hook)

    def __call__(self, model: nn.Module, img: Tensor, layer_name: str) -> dict:
        """
        Args:
            model (nn.Module): Input pre-trained model.
            img (np.ndarray): Image data.
            layer_name (str): The name of the target module of the model.
        
        Returns:
            tuple: (output, cam, prediction).
        """
        BS, _, H, W = img.size()
        # register model
        self._register_model(model, layer_name)

        # model prediction
        output = model(img)
        pred = torch.argmax(output, dim=-1)

        # saliency map computation
        y_c = output[torch.arange(BS), pred]
        data_hidden = self.data_hidden[0] # BS x C x H x W
        grads = torch.autograd.grad(y_c.sum(), data_hidden)[0] # BS x C x H x W

        # compute high-order grads
        grads_2 = grads ** 2
        grads_3 = grads_2 * grads
        sum_act = data_hidden.sum(dim=(2, 3), keepdim=True)
        aij = grads_2 / (2 * grads_2 + sum_act * grads_3 + 1e-8)
        aij = torch.where(grads != 0., aij, torch.zeros_like(aij))
        weights = (aij * torch.relu(grads)).sum(dim=(2, 3))
        cam =  torch.einsum("ij,ijml->iml", weights, data_hidden)
        
        cam = cam.clamp(min=0).unsqueeze(1)
        cam = F.upsample(cam, (H, W), mode="bilinear").squeeze()

        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }
