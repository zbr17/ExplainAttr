from typing import Tuple, Union, List
import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
from copy import deepcopy

import torch.nn as nn
from torch import Tensor
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from lime import lime_image

from .utils import norm_cam

class LIME:
    def __init__(self, num_sample, dataset, device, **kwargs):
        self.num_sample = num_sample
        self.dataset = dataset
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()
    
    def __call__(self, model: nn.Module, img: Tensor, layer_name: str) -> dict:
        # model preparation bs x C x H x W
        model.eval()

        # select class
        BS, C, H, W = img.size()
        output = model(img) # bs x K
        pred = torch.argmax(output, dim=-1)

        def predict(img):
            with torch.no_grad():
                img = torch.from_numpy(img).to(self.device)
                img = img.permute(0, 3, 1, 2).float()
                if self.dataset == "mnist":
                    img = img[:, 0:1, :, :]
                else:
                    img = img
                output = model(img)
            return output.detach().cpu().numpy().astype("double")

        # run lime
        with torch.no_grad():
            cam_list = []
            for i in range(BS):
                cur_img = img[i].permute(1, 2, 0).cpu().numpy().astype("double")
                cur_pred = pred[i].item()
                if cur_img.shape[-1] == 1:
                    cur_img = cur_img.squeeze()
                cur_exp = self.explainer.explain_instance(cur_img, predict, labels=cur_pred, hide_color=0, num_samples=self.num_sample, batch_size=100)

                # modified from https://nbviewer.org/github/kdd-lab/XAI-Survey/blob/main/Examples_Images.ipynb
                ind = cur_exp.top_labels[0]
                dict_heatmap = dict(cur_exp.local_exp[ind])
                heatmap = np.vectorize(dict_heatmap.get)(cur_exp.segments)
                cam_list.append(heatmap)

        cam = np.stack(cam_list)
        cam = torch.from_numpy(cam).to(self.device)

        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }