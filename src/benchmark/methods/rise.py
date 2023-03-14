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

class RISE:
    def __init__(self, num_mask, m_size, p, device, save_path="./data/rise_mask.ckpt", **kwargs):
        self.num_mask = num_mask
        self.m_size = m_size
        self.p = p
        self.device = device
        self.save_path = save_path

    def __call__(self, model: nn.Module, img: Tensor, layer_name: str) -> dict:
        model.eval()
        N = self.num_mask

        # select class
        BS, C, H, W = img.size()
        with torch.no_grad():
            output = model(img) # bs x K
            pred = torch.argmax(output, dim=-1)

            # generate masked img
            masks = self.generate_mask((H, W)) # N x 1 x H x W
            img_stacks = (masks.unsqueeze(1) * img.unsqueeze(0)) # N x BS x C x H x W

            scores = []
            for i in range(N):
                cur_out = model(img_stacks[i])
                cur_score = cur_out[torch.arange(BS), pred]
                scores.append(cur_score)
            scores = torch.stack(scores, dim=0) # N x BS
            cam = torch.einsum("ijk,im->mjk", masks.squeeze(), scores)
            cam = cam / (N * self.p)
        
        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }


    def generate_mask(self, data_size):
        m_size = self.m_size
        num_mask = self.num_mask

        masks = (torch.rand(num_mask, 1, m_size, m_size) < self.p).float().to(self.device)
        masks = F.upsample(masks, data_size, mode="bilinear")
        return masks