from typing import Tuple, Union, List
import numpy as np
from scipy.ndimage.interpolation import zoom

import torch.nn as nn
from torch import Tensor
import torch

def filter_bound(max_idx, h, w):
    mask = (max_idx // w != 0) & (max_idx // w != (h-1)) & (max_idx % w != 0 ) & (max_idx % w != (w-1))
    return mask

class TraceTopk:
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
        topk_num = 256
        scale = 3

        H, W = img.size(-2), img.size(-1)
        # register model
        self._register_model(model, layer_name)

        # trace back
        model.trace(True)
        output = model(img).squeeze(1)
        pred = torch.argmax(output, dim=-1).item()

        sample = self.data_hidden[0]
        data_hidden = sample.clone()
        bs, ch_n, h, w = data_hidden.size()
        data_hidden = data_hidden.view(bs, ch_n, -1)
        chm_data, max_idx = data_hidden.max(dim=-1)
        mask_bound = filter_bound(max_idx, h, w)
        chm_data[~mask_bound] = -100
        chm_topk_v, chm_topk = chm_data.topk(k=topk_num, dim=-1, largest=True)

        # version 1: construct the saliency map as a whole
        # mask_ch = torch.zeros(bs, ch_n).to(data_hidden.device)
        # mask_ch.scatter_(1, chm_topk, chm_topk_v)
        # mask_ch = mask_ch.view(bs, ch_n, 1)
        # mask_x = torch.zeros_like(data_hidden)
        # mask_x.scatter_(2, max_idx.unsqueeze(-1), 1)
        # data_hidden = data_hidden * mask_x * mask_ch
        # data_hidden = data_hidden.view(bs, ch_n, h, w) * scale

        # trace_out = model.trace_back(data_hidden, layer_name)
        # trace_zeros = model.trace_back(torch.zeros_like(data_hidden), layer_name)
        # trace_out = (trace_out - trace_zeros).abs().sum(dim=1)

        # trace_out = trace_out.cpu().squeeze().detach().numpy()
        # trace_out = np.maximum(trace_out, 0)
        # trace_out = zoom(trace_out, H / trace_out.shape[0])
        # trace_out = trace_out / (trace_out.max() + 1e-6)

        # version 2: construct the saliency map as the weighted sum of masks
        base_zeros = model.trace_back(torch.zeros_like(sample), layer_name)
        recon_img_list = []
        for idx in range(topk_num):
            tmp_data = torch.zeros_like(sample).view(bs, ch_n, -1)
            cur_chidx = chm_topk.flatten()[idx]
            cur_spidx = max_idx.flatten()[cur_chidx]
            tmp_data[0][cur_chidx][cur_spidx] = 1000
            tmp_recon = model.trace_back(x=tmp_data.view(bs, ch_n, h, w), module_name=layer_name)
            # generate masks
            tmp_recon = (tmp_recon - base_zeros).abs().sum(dim=1)
            tmp_recon = (tmp_recon > scale * tmp_recon.mean()).float()
            recon_img_list.append(tmp_recon)
        recon_img = torch.cat(recon_img_list, dim=0)
        ih, iw = recon_img.size(1), recon_img.size(2)
        trace_out = (chm_topk_v @ recon_img.view(topk_num, -1)).view(ih, iw)
        trace_out = trace_out / (trace_out.max() + 1e-6)
        trace_out = trace_out.detach().cpu().numpy()

        model.trace(False)
        return {
            "score": output,
            "cam": trace_out,
            "pred": pred
        }