from typing import Tuple, Union
import numpy as np
from scipy.ndimage.interpolation import zoom

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from .utils import norm_cam

class CAM:
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
        H, W = img.size(-2), img.size(-1)
        # register model
        self._register_model(model, layer_name)

        with torch.no_grad():
            # model prediction
            output = model(img)
            pred = torch.argmax(output, dim=-1)

            if hasattr(model, "fc"):
                # saliency map computation
                weights = model.fc.weight.data[pred, :]
                # weights = np.maximum(weights, 0)
                data_hidden = self.data_hidden[0]
                cam = torch.einsum("ij,ijml->iml", weights, data_hidden)
            else:
                data_hidden = self.data_hidden[0]
                BS, C, sH, sW = data_hidden.shape
                cam = data_hidden[torch.arange(BS), pred]
            cam = cam.clamp(min=0).unsqueeze(1)
            cam = F.upsample(cam, (H, W), mode="bilinear").squeeze()

        return {
            "score": output,
            "cam": norm_cam(cam),
            "pred": pred
        }
