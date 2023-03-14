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

import saliency.core as saliency

from .utils import norm_cam

class BasePAIR:
    def __init__(self, num_sample, dataset, device, **kwargs):
        self.num_sample = num_sample
        self.device = device
    
    @property
    def saliency(self):
        raise NotImplementedError()
    
    @property
    def get_mask_kwargs(self):
        raise NotImplementedError()
    
    def __call__(self, model: nn.Module, img: Tensor, layer_name: str) -> dict:
        # model preparation bs x C x H x W
        model.eval()

        # select class
        BS, C, H, W = img.size()
        output = model(img) # bs x K
        pred = torch.argmax(output, dim=-1)

        # modified from https://github.com/PAIR-code/saliency/blob/master/Examples_pytorch.ipynb
        class_idx_str = 'class_idx_str'
        def call_model_function(images, call_model_args=None, expected_keys=None):
            target_class_idx =  call_model_args[class_idx_str]
            images = torch.from_numpy(images).permute(0,3,1,2).float().to(self.device)
            images.requires_grad_(True)
            output = model(images)
            m = torch.nn.Softmax(dim=1)
            output = m(output)
            if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
                outputs = output[:,target_class_idx]
                grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))[0]
                grads = torch.movedim(grads, 1, 3)
                gradients = grads.detach().cpu().numpy()
                return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                raise NotImplementedError()
                one_hot = torch.zeros_like(output)
                one_hot[:,target_class_idx] = 1
                model.zero_grad()
                output.backward(gradient=one_hot, retain_graph=True)
                return conv_layer_outputs

        # run algorithm by saliency pkg
        cam_list = []
        for i in range(BS):
            call_model_args = {class_idx_str: pred[i]}
            cur_cam = self.saliency.GetMask(img[i].permute(1,2,0).cpu().numpy(), call_model_function, call_model_args, **self.get_mask_kwargs)
            cam_list.append(cur_cam)
        cam = np.stack(cam_list)
        if cam.ndim == 4:
            cam = cam.sum(axis=-1)
        cam = torch.from_numpy(cam).to(self.device)

        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }