from .tracable_module import TracableModule

from .simple_cnn import SimpleCNN
from .lenet import LeNet
from .lenet_conv import LeNetConv
from .lenet_fconv import LeNetFullConv
from .nomax_net_fconv import NomaxNetFullConv
from .resnet import resnet18, resnet50
from .vgg16 import vgg16
from .allconv12 import AllConv12

from . import layers

_factory_model = {
    "simple": SimpleCNN,
    "lenet": LeNet,
    "lenetc": LeNetConv,
    "lenetfc": LeNetFullConv,
    "nomaxnetfc": NomaxNetFullConv,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "vgg16": vgg16,
    "allconv12": AllConv12,
}

# nomax_net_fconv
out_list_dict = {
    "cifar10": {
        98: [16, 147, 427],
        97: [13, 99, 218],
        95: [10, 54, 89],
        90: [6, 21, 19],
        85: [4, 10, 10],
        80: [3, 5, 10]
    },
    "mnist": {
        98: [16, 219, 1133],
        97: [13, 163, 621],
        95: [11, 112, 385],
        90: [7, 50, 104],
        85: [6, 33, 52],
        80: [4, 20, 26],
    }
}

def _get_model_config(config):
    dataset = config.dataset
    model = config.model
    model_config = {}
    if dataset == "mnist":
        if "resnet" in model or "vgg" in model:
            model_config["num_classes"] = 10
        elif "allconv" in model:
            model_config["num_classes"] = 10
            model_config["setting"] = config.setting
            model_config["act_type"] = config.act_type
        elif "simple" in model:
            model_config["in_dim"] = 1
            model_config["in_size"] = 28
            model_config["act_type"] = config.act_type
        else:
            model_config["in_dim"] = 1
            model_config["in_size"] = 28
    elif dataset == "cifar10":
        if "resnet" in model or "vgg" in model:
            model_config["num_classes"] = 10
        elif "allconv" in model:
            model_config["num_classes"] = 10
            model_config["setting"] = config.setting
            model_config["act_type"] = config.act_type
        elif "simple" in model:
            model_config["in_dim"] = 3
            model_config["in_size"] = 32
            model_config["act_type"] = config.act_type
        else:
            model_config["in_dim"] = 3
            model_config["in_size"] = 32
    elif dataset == "imagenet":
        if "resnet" in model or "vgg" in model:
            model_config["num_classes"] = 1000
        elif "allconv" in model:
            model_config["num_classes"] = 1000
            model_config["setting"] = config.setting
            model_config["act_type"] = config.act_type
    
    if model == "nomaxnetfc":
        recon_ratio = int(100 * config.recon_ratio)
        # FIXME:
        model_config["out_list"] = [16, 64, 256] # out_list_dict[dataset][recon_ratio]
        print("FORCE let out-list = [16, 64, 256]")
    return model_config


def give_model(config):
    model = config.model
    model_config = _get_model_config(config)
    _out_object = _factory_model[model](**model_config)
    return _out_object