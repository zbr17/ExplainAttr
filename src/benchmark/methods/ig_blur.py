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
from ._base_saliency import BasePAIR

class BlurIG(BasePAIR):
    @property
    def saliency(self):
        return saliency.BlurIG()
    
    @property
    def get_mask_kwargs(self):
        return {
            "batch_size": 50
        }