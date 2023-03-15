#%%
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../src/record")))
import numpy as np
import h5py
import time
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from scipy.ndimage.interpolation import zoom
import argparse

from src.benchmark.misc import load_data_model, random_seed, absjoin, recover_data, preprocess_data
random_seed(42)
from src.benchmark.methods import give_method
from src.benchmark.methods.utils import np2tensor, tensor2np
from src.benchmark.metric import Metric, auc

import tbwriter

_vis_data_idx_ = {
    "mnist": [91, 23, 77, 52, 43, 48, 26, 31, 12, 8, 11, 38], # [8, 11, 38],
    "cifar10": [135, 130, 127, 77, 183, 63, 191, 164, 193, 160, 24, 36], # [160, 24, 36],
    "imagenet": list(range(36)) # [3, 7, 9]
}

class CONFIG:
    # NOTE: Model path
    _model_path_ = ...
    model = ...
    optim_name = ...
    act_type = ...    
    recon_ratio = 1.

    # core configs
    dataset = "imagenet"
    xai_name = "sam"
    nsteps = 100
    test_bs = 50
    total_num = 100
    use_softmax = False
    momen = None # 0.5
    reduction = "sum"
    line_type = ["deletion", "insertion"]

    _data_path_ = {
        "mnist": {
            "data": "~/Workspace/datasets/mnist"
        },
        "cifar10": {
            "data": "~/Workspace/datasets/cifar10"
        },
        "imagenet": {
            "data": "~/Workspace/datasets/imagenet/val",
        }
    }

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--xai_name", type=str, default=None)
parser.add_argument("--test_bs", type=int, default=None)
parser.add_argument("--momen", type=float, default=None)
parser.add_argument("--reduction", type=str, default=None)
parser.add_argument("--line_type", type=str, nargs="+", default=None)
opt = parser.parse_args(args=[])
# opt = parser.parse_args()
for k in dir(opt):
    if not k.startswith("_") and not k.endswith("_"):
        v = getattr(opt, k)
        if v is not None:
            print(f"Force config.{k} to {v}")
            setattr(CONFIG, k, v)

tbwriter.config(output_dir="~/Workspace/code/log/")
device = torch.device("cuda:0")
from_pretrain = CONFIG.dataset == "imagenet"
if CONFIG.dataset == "imagenet":
    CONFIG.step = 224 * 16
    CONFIG.metric_step = 224 * 8
    CONFIG.layer_name = "layer4[2].bn3"
    # CONFIG.layer_name = "features[30]"
    CONFIG.klen = 31
    CONFIG.ksig = 5
    CONFIG.dataset = "imagenet"
    CONFIG.model = "resnet50"
    # CONFIG.model = "vgg16"
    CONFIG.optim_name = None
    CONFIG.act_type = None
elif CONFIG.dataset == "mnist" or CONFIG.dataset == "cifar10":
    CONFIG.step = 10
    CONFIG.metric_step = 10
    CONFIG.layer_name = "act3"
    CONFIG.klen = 11
    CONFIG.ksig = 5
    if CONFIG.dataset == "mnist":
        mpath = "~/Workspace/code/log/mnist-sgd"
    elif CONFIG.dataset == "cifar10":
        mpath = "~/Workspace/code/log/cifar10-sgd"
    CONFIG._model_path_ = mpath
    # Read config file
    config_dict = yaml.load(open(os.path.join(CONFIG._model_path_, "config.yaml")), Loader=yaml.FullLoader)
    CONFIG.dataset = config_dict["dataset"]
    CONFIG.model = config_dict["model"]
    CONFIG.optim_name = config_dict["optim"]
    CONFIG.act_type = config_dict.get("act_type", "leaky")
else:
    raise KeyError(f"Invalid dataset {CONFIG.dataset}!")


for k in dir(CONFIG):
    if not k.startswith("_") and not k.endswith("_"):
        print(f"{k}: {getattr(CONFIG, k)}")
data, label, mapping, model = load_data_model(CONFIG, is_check=False, from_pretrain=from_pretrain, total_num=CONFIG.total_num)
model = model.to(device)
model.eval()
if data.ndim == 4 and data.shape[-1] != 3:
    data = data.transpose(0, 2, 3, 1)
def show_samples(config, data, label, mapping):
    data_name = config.dataset
    vis_idx_list = _vis_data_idx_[data_name]
    num = len(vis_idx_list)
    data = recover_data(data)
    _, ax = plt.subplots(num//4, 4, figsize=(3*4, 3*num//4))
    for i in range(num//4):
        for j in range(4):
            cur_ax = ax[i][j]
            cur_ax.axis("off")
            idx = vis_idx_list[i*4+j]
            label_str = mapping[label[idx]]
            if data.ndim == 3:
                cur_ax.imshow(data[idx, :], cmap="gray")
            elif data.ndim == 4:
                cur_ax.imshow(data[idx, :])
            cur_ax.set_title(label_str)
    return vis_idx_list
vis_idx_list = show_samples(CONFIG, data, label, mapping)

#%%
### Show demos
xai_obj = give_method(
    CONFIG,
    line_type=CONFIG.line_type,             # SAM / naive: ["deletion", "insertion"]
    reduction=CONFIG.reduction,             # SAM / naive
    step=CONFIG.step,                       # SAM / naive
    klen=CONFIG.klen,                       # SAM / naive
    ksig=CONFIG.ksig,                       # SAM / naive
    momen=CONFIG.momen,                     # SAM / naive
    n_frag=50,                              # SAM
    nsteps=CONFIG.nsteps,                   # IG
    num_mask=400,                           # RISE
    m_size=8,                               # RISE
    p=0.1,                                  # RISE
    num_grad=100,                           # SG
    noise_level=0.3,                        # SG
    num_sample=100,                         # LIME
    dataset=CONFIG.dataset,                 # LIME
    model=model,                            # LRP
    device=device,
)
layer_name = CONFIG.layer_name

sub_label, sub_img = label[vis_idx_list], data[vis_idx_list]
start = time.time()
input = np2tensor(sub_img).to(device)
output_dict = xai_obj(model, input, layer_name)
pred = output_dict["pred"]
score = torch.softmax(output_dict["score"], dim=-1)
score = score[torch.arange(score.size(0)), pred]
score = tensor2np(score)
mask_all = tensor2np(output_dict["cam"])
end = time.time()
print(f"Running time: {end - start}s")

for i, idx in enumerate(vis_idx_list):
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    sub_label, sub_img, mask = label[idx], data[idx], mask_all[i]
    sub_img = recover_data(sub_img, is_batch=False)
    ax[0].imshow(sub_img, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title(f"score: {float(score[i]):.4f}")
    ax[0].title.set_size(30)
    ax[1].imshow(mask, cmap="gray")
    ax[1].axis("off")
    ax[2].imshow(sub_img, alpha=0.9, cmap="gray")
    ax[2].imshow(mask, alpha=0.4, cmap="jet")
    ax[2].axis("off")

#%%
### Compute score
# get the image indices
img_idx_list = np.random.permutation(len(data))[:100].reshape(-1, CONFIG.test_bs)
step_per = CONFIG.metric_step
metric = Metric(step_per=step_per, device=device, use_softmax=CONFIG.use_softmax, klen=CONFIG.klen, ksig=CONFIG.ksig)

del_list, ins_list = [], []
for idx in tqdm(img_idx_list):
    sub_img, sub_label = data[idx], label[idx]
    input = np2tensor(sub_img).to(device)
    metric_dict = metric(model, input, sub_label, layer_name, xai_obj)
    del_list.append(metric_dict["deletion"]["auc"])
    ins_list.append(metric_dict["insertion"]["auc"])
del_list = np.concatenate(del_list)
ins_list = np.concatenate(ins_list)

for k in dir(CONFIG):
    if not k.startswith("__") and not k.endswith("__"):
        print(f"{k}: {getattr(CONFIG, k)}")
print(f"Deletion: mean={del_list.mean()}, std={del_list.std()}")
print(f"Insertion: mean={ins_list.mean()}, std={ins_list.std()}")