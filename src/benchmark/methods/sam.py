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

def batch_l1_dist(input1, input2=None):
    bs = input1.size(0)
    if input2 is None:
        diff = torch.abs(input1).view(bs, -1)
    else:
        diff = torch.abs(input1 - input2).view(bs, -1)
    return torch.sum(diff, dim=-1)

class SAM:
    def __init__(self, line_type, reduction, step=5, n_frag=5, klen=11, ksig=5, momen=None, **kwargs):
        self.step = step
        self.n_frag = n_frag
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
            # return torch.zeros_like(img)
            return gaussian_blur(img, klen=self.klen, ksig=self.ksig)
        else:
            raise KeyError(f"Invalid input line-type: {line_type}!")

    def generate_cam(self, model, pred, img, ref_point, line_type):
        # set internal states
        self._check_state(line_type)
        BS, C, H, W = img.shape
        step = self.step
        
        # initiate variables
        cur_point = img.clone() if self.is_inverse else ref_point
        end_point = ref_point if self.is_inverse else img.clone()
        total_l1_dist = batch_l1_dist(cur_point, end_point)
        step_bound = total_l1_dist / self.n_frag
        cam = torch.zeros(BS, H, W).to(cur_point.device)

        # recursively build line integral
        grads = None
        # initiate selected_mask
        selected_mask = ~ torch.isclose(cur_point, end_point)
        selected_mask = torch.any(selected_mask, dim=1) # BS x H x W
        while torch.any(selected_mask):
            # compute batch_mask
            batch_mask = torch.any(selected_mask.view(BS, -1), dim=-1) # BS
            N = torch.sum(batch_mask)
            selected_mask = selected_mask[batch_mask]

            # obtain score
            sub_cur_point = cur_point[batch_mask] # N x C x H x W
            sub_cur_point.requires_grad_(True)
            scores = model(sub_cur_point) # N x K
            scores = scores[torch.arange(N), pred[batch_mask]] # N x K

            # compute gradients
            actual_grads = autograd.grad(torch.sum(scores), sub_cur_point)[0] # N x C x H x W
            if (grads is not None) and (self.momen is not None):
                grads[batch_mask] = (self.momen * actual_grads + (1 - self.momen) * grads[batch_mask]).detach()
            else:
                grads = torch.empty_like(cur_point)
                grads[batch_mask] = actual_grads.detach()
            sub_grads = grads[batch_mask]

            with torch.no_grad():
                # compute direction
                delta_direc = end_point[batch_mask] - sub_cur_point.detach() # N x C x H x W
                projection = torch.sum(sub_grads * delta_direc, dim=1)  # N x H x W

                q = step / (H*W)
                if self.is_inverse:
                    projection[~ selected_mask] = float("inf")
                    cur_quantile = torch.quantile(projection.view(N, -1), q=q, dim=-1, interpolation="lower", keepdim=True) + 1e-6
                    update_mask = projection < cur_quantile[:, None]
                else:
                    projection[~ selected_mask] = float("-inf")
                    cur_quantile = torch.quantile(projection.view(N, -1), q=1-q, dim=-1, interpolation="higher", keepdim=True) - 1e-6
                    update_mask = projection > cur_quantile[:, None]

                # compute next position
                update_mask = update_mask.unsqueeze(1).expand(-1, C, -1, -1) # N x C x H x W
                move_full_step = delta_direc * update_mask # N x C x H x W
                move_l1_dist = batch_l1_dist(move_full_step) # N

                # adjust l1-distance when outranges
                is_outrange = move_l1_dist > step_bound[batch_mask]
                adjust_l1_ratio = step_bound[batch_mask] / (move_l1_dist + 1e-8)
                sub_cur_point = sub_cur_point.detach()
                move_full_step[is_outrange] *= adjust_l1_ratio[is_outrange, None, None, None]
                
                # update current position and cam
                cur_point[batch_mask] += move_full_step
                cam[batch_mask] += torch.sum(move_full_step * sub_grads, dim=1)

                # update selected_mask
                selected_mask = ~ torch.isclose(cur_point, end_point)
                selected_mask = torch.any(selected_mask, dim=1) # BS x H x W

                move_norm = torch.norm(move_full_step.view(N, -1), p=2, dim=-1)
                cur_p_norm = torch.norm(cur_point.view(BS, -1), p=2, dim=-1)
                tbwriter.add_scalar("mean_move_norm", move_norm.mean())
                tbwriter.add_scalar("mean_cur_norm", cur_p_norm.mean())
                tbwriter.add_scalar("max_cam", -cam.min())

            # NOTE: debug
            # try:
            #     pi += 1
            # except:
            #     pi = 0
            # plot_img(cam[-2] / 1.2, cur_point[-2], f"tests_samp/1_1_veri/cam_{pi}.png", alpha=0.9)
            # plot_img(cam[-2] / 1.2, cur_point[-2], f"tests_samp/1_1_veri/img_{pi}.png", alpha=0.0)

        cam = - cam if self.is_inverse else cam
        return cam

def plot_img(mask, img, title: str, alpha=0.5):
    from src.benchmark.misc import recover_data
    mask = mask.clamp(min=0, max=1)
    mask = mask.cpu().numpy()
    img = recover_data(img, is_batch=False)
    img = img.cpu().numpy()
    _, ax = plt.subplots(1, 1)
    ax.imshow(img)
    ax.imshow(mask, alpha=alpha, cmap="gist_heat")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{title}", dpi=500)