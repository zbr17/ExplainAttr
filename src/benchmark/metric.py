import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
import io
import os
from PIL import Image
import time

import torch.nn as nn
from torch import Tensor
import torch

import tbwriter
from .methods.utils import np2tensor, tensor2np
from .misc import recover_data

_VIS_PLOT_ = False

def auc(data: Tensor, mtype: str):
    if mtype == "del":
        data_peak = data[:, 0].unsqueeze(-1)
    elif mtype == "ins":
        data_peak = data[:, -1].unsqueeze(-1)
    data = data / data_peak
    area = data.sum(dim=-1) / data.size(-1)
    return tensor2np(area)

def gkern(ch, klen, ksig):
    inp = np.zeros((klen, klen))
    inp[klen//2, klen//2] = 1
    k = gaussian_filter(inp, ksig)
    kern = np.zeros((ch, ch, klen, klen))
    for i in range(ch):
        kern[i, i] = k
    return torch.from_numpy(kern.astype("float32"))

def gaussian_blur(img, klen, ksig):
    device = img.device
    ch = img.size(1)
    # get gkern
    kern = gkern(ch, klen, ksig).to(device)
    # compute gaussian blur
    img_out = nn.functional.conv2d(img, kern, padding=klen//2)
    return img_out

class Metric:
    def __init__(self, step_per: int = 10, klen: int = 11, ksig: int = 5, device=torch.device("cpu"), use_softmax=False):
        self.step_per = step_per
        self.device = device
        self.klen = klen
        self.ksig = ksig
        self.use_softmax = use_softmax

        self.saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]
        self.comp = compress_jpeg().to(device)

    def __call__(self, model: nn.Module, img: np.ndarray, label: int, layer_name: str, explainer):
        """
        Args:
            model (nn.Module): Deep artificial neural network.
            img (np.ndarray): Images to be tested.
            label (int): Image label.
            explainer (object): Explainable method.
        """
        BS, C, H, W = img.shape
        self.nsteps = (H * W + self.step_per - 1) // self.step_per

        # Generate prediction and saliency map
        output_dict = explainer(model, img, layer_name=layer_name)
        mask = output_dict["cam"]
        pred = output_dict["pred"]
        output = {}

        # Compute deletion
        deletion_score = self._compute_scores(
            start_sample=img.clone(),
            end_sample=torch.zeros_like(img),
            model=model,
            mask=mask,
            pred=pred
        )
        output["deletion"] = {
            "auc": auc(deletion_score, "del"),
            "score": deletion_score
        }

        # Compute insertion
        insertion_score = self._compute_scores(
            start_sample=gaussian_blur(img, self.klen, self.ksig),
            end_sample=img.clone(),
            model=model,
            mask=mask,
            pred=pred
        )
        output["insertion"] = {
            "auc": auc(insertion_score, "ins"),
            "score": insertion_score
        }

        return output
    
    def _compute_scores(
            self, 
            start_sample: Tensor, 
            end_sample: Tensor, 
            model: nn.Module, 
            mask: Tensor, 
            pred: int
        ):
        nsteps = self.nsteps
        BS, C, H, W = start_sample.shape
        step = self.step_per

        end_sample = end_sample.reshape(BS, C, H*W)
        scores = torch.zeros(BS, nsteps + 1).to(self.device)
        mask = mask.view(BS, -1)
        _, sort_order = torch.sort(mask, dim=-1, descending=True)
        with torch.no_grad():
            for i in range(nsteps + 1):
                output = model(start_sample)
                if self.use_softmax:
                    output = torch.softmax(output, dim=-1)
                cur_score = output[torch.arange(BS), pred]
                scores[:, i] = cur_score
                if _VIS_PLOT_: 
                    max_v = output.max(dim=-1)[0]
                    tbwriter.log.add_histogram(f"max_v", max_v, global_step=i)
                    tbwriter.log.add_histogram(f"pred_v", cur_score, global_step=i)
                if i < nsteps:
                    cur_ords = sort_order[:, (step*i) : (step*(i+1))]
                    cur_ords = cur_ords.unsqueeze(1).expand(BS, C, -1)
                    start_sample = start_sample.reshape(BS, C, H*W)
                    start_sample.scatter_(2, cur_ords, end_sample.gather(2, cur_ords))
                    start_sample = start_sample.reshape(BS, C, H, W)
        
        return scores.detach()

