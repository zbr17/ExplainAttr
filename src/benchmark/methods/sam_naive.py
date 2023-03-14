from typing import Tuple, Union, List
import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

import torch.nn as nn
from torch import Tensor
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from ..metric import gaussian_blur
from .utils import norm_cam

import tbwriter

def plot(data, title: str, H, W):
    data = data.view(H, W)
    data = (data - data.min()) / (data.max() - data.min() + 1e-6)
    data = data.cpu().numpy()
    _, ax = plt.subplots(1, 1)
    ax.imshow(data, cmap="jet")
    plt.savefig(f"test_tmp/{title}.png")

def plot_img(data, img, title: str, H, W):
    data = data.view(H, W).clone()
    data = data.cpu().numpy()
    img = img.cpu().numpy()
    data = (img * (data==0)).squeeze() # .transpose(1, 2, 0)
    data = (data - data.min()) / (data.max() - data.min() + 1e-6)
    _, ax = plt.subplots(1, 1)
    ax.imshow(data)
    plt.savefig(f"test_tmp2/{title}.png")


class SAMNaive:
    def __init__(self, line_type, reduction, step=5, klen=11, ksig=5, momen=None, **kwargs):
        self.step = step
        self.line_pool = line_type
        self.reduction = reduction
        self.klen = klen
        self.ksig = ksig
        self.momen = momen

    def _check_state(self, line_type):
        if line_type in ["deletion"]:
            self.is_inverse = True
        elif line_type in ["insertion"]:
            self.is_inverse = False
        else:
            raise KeyError(f"Invalid input line-type: {line_type}!")

    def __call__(self, model: nn.Module, img: Tensor, layer_name: str) -> dict:
        # model preparation bs x C x H x W
        model.eval()

        # select class
        _, _, H, W = img.size()
        output = model(img) # bs x K
        pred = torch.argmax(output, dim=-1)

        # compute cam
        cam_list = []
        for line_type in self.line_pool:
            ref_point = self.generate_ref_point(img, line_type)
            cam = self.generate_cam(model, pred, img, ref_point, line_type)
            cam = self.postproc_cam(cam)
            cam_list.append(cam)
        cam = torch.stack(cam_list).sum(dim=0)

        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }
    
    def postproc_cam(self, cam):
        if self.reduction == "sum":
            return cam
        elif self.reduction == "norm":
            return norm_cam(cam)
        else:
            raise KeyError(f"Invalid reduction type: {self.reduction}!")

    def generate_ref_point(self, img, line_type):
        if line_type == "deletion":
            return torch.zeros_like(img)
        elif line_type == "insertion":
            return gaussian_blur(img, klen=self.klen, ksig=self.ksig)
        else:
            raise KeyError(f"Invalid input line-type: {line_type}!")

    def generate_cam(self, model, pred, img, ref_point, line_type):
        # set internal states
        self._check_state(line_type)
        BS, C, H, W = img.shape
        
        # initiate variables
        cur_point = img.clone() if self.is_inverse else ref_point
        end_point = ref_point if self.is_inverse else img.clone()
        cam = torch.zeros(BS, H, W).to(cur_point.device)

        # recursively build line integral
        step = self.step
        nsteps = int((H*W + self.step - 1) // step)
        grads = None
        for i in range(nsteps):
            # obtain score
            cur_point.requires_grad_(True)
            scores = model(cur_point)
            scores = scores[torch.arange(BS), pred] # BS x 1

            # compute gradients
            actual_grads = autograd.grad(torch.sum(scores), cur_point)[0] # BS x C x H x W
            if (grads is not None) and (self.momen is not None):
                grads = (self.momen * actual_grads + (1 - self.momen) * grads).detach()
            else:
                grads = actual_grads.detach()

            # compute optimal projection
            with torch.no_grad():
                # compute selected mask
                selected_mask = ~ torch.isclose(cur_point, end_point)
                if not torch.any(selected_mask): break
                delta_direc = end_point - cur_point.detach()
                selected_mask = torch.any(selected_mask, dim=1) # BS x H x W
                projection = torch.sum(grads * delta_direc, dim=1)  # BS x H x W

                q = step / (H*W)
                if self.is_inverse:
                    projection[~ selected_mask] = float("inf")
                    cur_quantile = torch.quantile(projection.view(BS, -1), q=q, dim=-1, interpolation="lower", keepdim=True) + 1e-6
                    update_mask = projection < cur_quantile[:, None]
                else:
                    projection[~ selected_mask] = float("-inf")
                    cur_quantile = torch.quantile(projection.view(BS, -1), q=1-q, dim=-1, interpolation="higher", keepdim=True) - 1e-6
                    update_mask = projection > cur_quantile[:, None]

                # compute next position and cam
                update_mask = update_mask.unsqueeze(1).expand(-1, C, -1, -1)
                move_full_step = delta_direc * update_mask
                cur_point = cur_point.detach()
                cur_point += move_full_step
                cam += torch.sum(move_full_step * grads, dim=1)

            # NOTE: debug
            # plot(cam, f"{i}", H, W)
            # plot_img(order, img, f"{i}", H, W)

        cam = - cam if self.is_inverse else cam
        return cam