from typing import Tuple, Union, List
import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

import torch.nn as nn
from torch import Tensor
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from .utils import norm_cam

class SG:
    def __init__(self, num_grad, noise_level, device, **kwargs):
        self.num_grad = num_grad
        self.noise_level = noise_level
        self.device = device

    def __call__(self, model: nn.Module, img: Tensor, layer_name: str) -> dict:
        model.eval()
        N = self.num_grad
        BS, C, H, W = img.size()

        img_flat = img.view(BS, -1)
        sigma = self.noise_level * (img_flat.max(dim=-1)[0] - img_flat.min(dim=-1)[0])
        sigma = sigma.view(BS,1,1,1).repeat(1,C,H,W)

        # select class
        output = model(img) # bs x K
        pred = torch.argmax(output, dim=-1)

        smooth_grads = torch.zeros_like(img)
        for i in range(N):
            # generate noise
            noise = torch.normal(mean=0, std=sigma**2)
            img_noised = img + noise
            img_noised.requires_grad_(True)
            scores = model(img_noised)
            scores = scores[torch.arange(BS), pred]

            # compute gradients
            grads = autograd.grad(torch.sum(scores), img_noised)[0]
            smooth_grads += grads.detach()
        
        smooth_grads = smooth_grads / N
        cam = torch.sum(smooth_grads, dim=1)
        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }
        