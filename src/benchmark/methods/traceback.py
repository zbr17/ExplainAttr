from typing import Tuple, Union, List
import numpy as np
from scipy.ndimage.interpolation import zoom

import torch.nn as nn
from torch import Tensor
import torch

class TraceBack:
    def _register_model(self, model: nn.Module, layer_name: str):
        self.data_hidden = []
        def hook(module, input, output):
            self.data_hidden.append(output)
        layer: nn.Module = getattr(model, layer_name)
        layer.register_forward_hook(hook)

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

        # trace back
        model.trace()
        output = model(img).squeeze(1)
        pred = torch.argmax(output, dim=-1).item()
        data_hidden = self.data_hidden[0]
        trace_out = model.trace_back(data_hidden, layer_name)

        trace_out = trace_out.cpu().squeeze().detach().numpy()
        trace_out = np.maximum(trace_out, 0)
        trace_out = zoom(trace_out, H / trace_out.shape[0])
        trace_out = trace_out / (trace_out.max() + 1e-6)
        return {
            "score": output,
            "cam": trace_out,
            "pred": pred
        }