from torch.utils.tensorboard import SummaryWriter
import functools
import sys
import os
from collections import defaultdict
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn

###########CONFIG##############
_DETAILED = False
###############################

_step_count_dict: dict = defaultdict(int)
_vis_conv: bool = False
_visualizer = None

class DefaultLog:
    pass

global log
log = DefaultLog()

_func_list_ = ['add_scalar', 'add_scalars', 'add_histogram', 'add_image', 
               'add_images', 'add_figure', 'add_video', 'add_audio', 'add_text', 
               'add_graph', 'add_embedding', 'add_hparams']

def do_nothing(*args, **kwargs):
    return

@functools.lru_cache()
def config(output_dir, dist_rank=0, only_main_rank=True):
    global log
    if only_main_rank and (dist_rank != 0):
        for _func_name in _func_list_:
            setattr(log, _func_name, do_nothing)
    else:
        log_dir = os.path.join(output_dir, f'Rank{dist_rank}')
        log = SummaryWriter(
            log_dir=log_dir
        )

def set_visualizer(visualizer):
    global _visualizer
    _visualizer = visualizer

def count_step(tag: str) -> int:
    global _step_count_dict
    _step_count_dict[tag] += 1
    return _step_count_dict[tag] - 1

def add_scalar(tag: str, value: Union[Tensor, float, int]):
    curr_step = count_step(tag)
    log.add_scalar(tag, value, global_step=curr_step)

### handle statistic visualization

def is_vis_conv() -> bool:
    return _vis_conv

def set_vis_conv(is_vis: bool = True):
    global _vis_conv
    _vis_conv = is_vis

def update_conv_vars(module: nn.Module, values: Tensor, tag: str =""):
    """
    Count the (defailed) feature maps distribution before / after adding the bias parameters.
    """
    if _vis_conv:
        with torch.no_grad():
            register_dict = _visualizer.register_dict
            module_name = register_dict[id(module)]
            tag = f"{module_name}.{tag}"
            if _DETAILED:
                for i in range(values.size(-1)):
                    sub_tag = tag + "/" + str(i)
                    step = count_step(sub_tag)
                    sub_values = values[:, :, i]
                    log.add_histogram(sub_tag, sub_values, global_step=step)

                    nonzero_ratio = torch.sum(sub_values > 0) / torch.numel(sub_values)
                    sub_tag = tag + ".ratio/" + str(i)
                    step = count_step(sub_tag)
                    log.add_scalar(sub_tag, nonzero_ratio, step)
            else:
                step = count_step(tag)
                log.add_histogram(tag, values, global_step=step)

def update_conv_norm(module: nn.Module, values: Tensor, tag: str=""):
    """
    Count the convolution kernel norms.
    """
    if _vis_conv:
        with torch.no_grad():
            register_dict = _visualizer.register_dict
            module_name = register_dict[id(module)]
            values = values.t() @ values
            out_d = values.size(0)
            diag_mask = torch.eye(out_d).to(values.device).bool()
            indiag_mask = ~ diag_mask
            # diag
            sub_tag = f"{module_name}.{tag}.diag"
            step = count_step(sub_tag)
            sub_values = values[diag_mask]
            log.add_histogram(sub_tag, sub_values, global_step=step)
            sub_tag += ".mean"
            step = count_step(sub_tag)
            log.add_scalar(sub_tag, torch.mean(sub_values), global_step=step)

            # indiag
            sub_tag = f"{module_name}.{tag}.indiag"
            step = count_step(sub_tag)
            sub_values = values[indiag_mask]
            log.add_histogram(sub_tag, sub_values, global_step=step)
            sub_tag += ".mean"
            step = count_step(sub_tag)
            log.add_scalar(sub_tag, torch.mean(sub_values), global_step=step)

def update_conv_val_norm(module: nn.Module, values: Tensor, tag: str=""):
    """
    Count the convolution input norms
    """
    if _vis_conv:
        with torch.no_grad():
            register_dict = _visualizer.register_dict
            module_name = register_dict[id(module)]
            tag = f"{module_name}.{tag}"
            step = count_step(tag)
            norm_value = values.detach().norm(dim=1)
            log.add_histogram(tag, norm_value, global_step=step)

