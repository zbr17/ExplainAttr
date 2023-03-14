import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import trange, tqdm
import math
import random
from PIL import Image

from torchvision.datasets import MNIST, CIFAR10, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from src.bogd.models import give_model
from src.benchmark import models

_BS_ = 50

def absjoin(*args):
    return os.path.abspath(os.path.join(*args))

def random_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def _get_all_iter_data(data_loader, iterations=None):
    data_list, label_list = [], []
    for idx, (data, label) in enumerate(tqdm(data_loader)):
        data_list.append(data)
        label_list.append(label)
        if (iterations is not None) and (idx >= iterations):
            break
    data_list = torch.cat(data_list, dim=0).squeeze(1).numpy()
    label_list = torch.cat(label_list, dim=0).numpy()
    return data_list, label_list

def load_data_model(config, is_check=False, from_pretrain=False, total_num=100):
    # load data
    data_name = config.dataset
    _data_path_ = config._data_path_
    _model_path_ = config._model_path_
    if data_name == "mnist":
        test_src = MNIST(root=_data_path_["mnist"]["data"], transform=transforms.ToTensor(), train=False, download=True)
        test_loader = DataLoader(test_src, batch_size=_BS_, shuffle=True, drop_last=False)
        data, label = _get_all_iter_data(test_loader)
        mapping = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif data_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        test_src = CIFAR10(root=_data_path_["cifar10"]["data"], train=False, transform=transform, download=True)
        test_loader = DataLoader(test_src, batch_size=_BS_, shuffle=False, drop_last=False)
        data, label = _get_all_iter_data(test_loader)
        mapping = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif data_name == "imagenet":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        test_src = ImageFolder(root=_data_path_["imagenet"]["data"], transform=transform)
        test_loader = DataLoader(test_src, batch_size=_BS_, shuffle=True, drop_last=False)
        data, label = _get_all_iter_data(test_loader, np.ceil(total_num/_BS_))
        mapping = [str(idx) for idx in range(1000)]
    else:
        raise KeyError(f"Invalid dataset name: {data_name}!")
    
    # load model
    if from_pretrain:
        model = getattr(models, config.model)(pretrained=True)
    else:
        model = give_model(config)
        param_path = os.path.join(_model_path_, f"model.ckpt")
        params = torch.load(param_path, map_location="cpu")["model"]
        model.load_state_dict(params)
    model.eval()

    # check
    if is_check:
        bs = _BS_
        acc, count = 0, 0
        epochs = math.floor(len(data) / bs)
        for i in trange(epochs):
            s, e = bs * i, bs * (i+1)
            data_slice = torch.from_numpy(data[s:e])
            if data_slice.ndim == 3:
                data_slice = data_slice.unsqueeze(1)
            out = model(data_slice)
            pred = torch.argmax(out, dim=-1).numpy()
            acc += np.sum(pred == label[s:e])
            count += bs
        print(f"Model prediction accuracy: {acc / count}")
    return data, label, mapping, model

def recover_data(data, is_batch=True):
    ndim_flag = 3 if is_batch else 2
    size_list = [1]*ndim_flag + [3]
    if data.ndim == ndim_flag: # MNIST
        data = data
    elif data.ndim == ndim_flag+1:
        if data.shape[-1] != 3:
            if isinstance(data, torch.Tensor):
                if data.ndim == 3:
                    data = data.permute(1, 2, 0)
                elif data.ndim == 4:
                    data = data.permute(0, 2, 3, 1)
            else:
                if data.ndim == 3:
                    data = data.transpose(1, 2, 0)
                elif data.ndim == 4:
                    data = data.transpose(0, 2, 3, 1)
        if data.shape[1] == 32: # CIFAR-10
            data = 0.5 * data + 0.5
        elif data.shape[1] == 224: # ImageNet
            inv_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(*size_list)
            inv_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(*size_list)
            if isinstance(data, torch.Tensor):
                device = data.device
                inv_mean = torch.from_numpy(inv_mean).to(device)
                inv_std = torch.from_numpy(inv_std).to(device)
            data = data * inv_std + inv_mean
    else:
        raise TypeError(f"Invalid ndim of images: {data.ndim}")
    return data

def preprocess_data(data, is_batch=True):
    ndim_flag = 3 if is_batch else 2
    size_list = [1]*ndim_flag + [3]
    if data.ndim == ndim_flag: # MNIST
        data = data
    elif data.ndim == ndim_flag+1:
        if data.shape[-1] != 3:
            if isinstance(data, torch.Tensor):
                if data.ndim == 3:
                    data = data.permute(1, 2, 0)
                elif data.ndim == 4:
                    data = data.permute(0, 2, 3, 1)
            else:
                if data.ndim == 3:
                    data = data.transpose(1, 2, 0)
                elif data.ndim == 4:
                    data = data.transpose(0, 2, 3, 1)
        if data.shape[1] == 32: # CIFAR-10
            data = 2 * (data - 0.5)
        elif data.shape[1] == 224: # ImageNet
            inv_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(*size_list)
            inv_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(*size_list)
            if isinstance(data, torch.Tensor):
                device = data.device
                inv_mean = torch.from_numpy(inv_mean).to(device)
                inv_std = torch.from_numpy(inv_std).to(device)
            data = (data - inv_mean) / inv_std
    else:
        raise TypeError(f"Invalid ndim of images: {data.ndim}")
    return data

if __name__ == "__main__":
    class CONFIG:
        dataset = "mnist"
        model = "simple"
        optim_name = "sgd"
    data, label, mapping, model = load_data_model(CONFIG)
    print(data.shape)
    print(label.shape)
    print(mapping)
    print(model)
