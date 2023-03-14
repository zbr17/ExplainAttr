from typing import Tuple, Union
import numpy as np
from scipy.ndimage.interpolation import zoom

import torch.nn as nn
from torch import Tensor
import torch
import torch.autograd as autograd

from .utils import norm_cam

class IG:
    def __init__(self, nsteps, **kwargs):
        self.nsteps = nsteps
    
    def __call__(self, model: nn.Module, img: Tensor, layer_name: str = None) -> dict:
        # model preparation
        BS, C, H, W = img.shape
        nsteps = self.nsteps
        model.eval()

        # select class
        output = model(img).squeeze(1) # bs x K
        pred = torch.argmax(output, dim=-1)

        # initialization
        cur_point = img.clone()
        ref_point = torch.zeros_like(cur_point) # NOTE: version 1: zero image as reference image
        step_direc = (ref_point - cur_point) / nsteps
        cam = torch.zeros_like(cur_point)

        # recursively build line integral
        for i in range(nsteps):
            # obtain score
            cur_point.requires_grad_(True)
            scores = model(cur_point)[torch.arange(BS), pred] # BS x 1
            # compute gradients
            grads = autograd.grad(torch.sum(scores), cur_point)[0] # BS x C x H x W
            cur_point = cur_point.detach()
            # accumulate cam
            cam += grads * step_direc
            # update cur-point
            cur_point += step_direc
        
        cam = - cam.sum(dim=1)
        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }