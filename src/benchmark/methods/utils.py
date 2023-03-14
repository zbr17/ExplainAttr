from typing import Tuple
import numpy as np

from torch import Tensor
import torch

_CLAMP_NEG_ = False

def norm_cam(cam: Tensor):
    if _CLAMP_NEG_:
        cam = cam.clamp(min=0)
    BS, H, W = cam.size()
    cam = cam.view(BS, H*W)
    max_v = cam.max(dim=-1)[0].view(BS, 1)
    min_v = cam.min(dim=-1)[0].view(BS, 1)
    cam = (cam - min_v) / (max_v - min_v + 1e-8)
    cam = cam.view(BS, H, W)
    return cam

def np2tensor(img: Tuple[np.ndarray, Tensor]):
    """
    img: BS x [C x] x H x W
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    
    if img.ndim == 3:
        img = img.unsqueeze(1)
    elif img.ndim == 4:
        pass
    else:
        raise TypeError(f"Invalid ndim of img: {img.ndim}")
    
    if img.size(-1) == 3:
        img = img.permute(0, 3, 1, 2)

    return img

def tensor2np(img: Tuple[np.ndarray, Tensor]):
    """
    img: BS x [C x] x H x W
    """
    if isinstance(img, Tensor):
        if img.requires_grad:
            img = img.detach()
        img = img.cpu().numpy()
    elif isinstance(img, np.ndarray):
        pass
    else:
        raise TypeError(f"Wrong input type: {type(img)}!")

    return img