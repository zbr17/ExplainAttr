import torch
import torchvision
from .conv       import Conv2d 
from .linear     import Linear
from .sequential import Sequential, Bottleneck

conversion_table = { 
        'Linear': Linear,
        'Conv2d': Conv2d
    }

# # # # # Convert torch.models.resnetxx to lrp model
def convert_resnet(module, modules=None):
    # First time
    if modules is None: 
        modules = []
        for m in module.children():
            convert_resnet(m, modules=modules)
            
            # if isinstance(m, torch.nn.Sequential):
            #     break
            
            # Vgg model has a flatten, which is not represented as a module
            # so this loop doesn't pick it up.
            # This is a hack to make things work.
            if isinstance(m, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)): 
                modules.append(torch.nn.Flatten())

        sequential = Sequential(*modules)
        return sequential

    # Recursion
    if isinstance(module, torch.nn.Sequential): 
        for m in module.children():
            convert_resnet(m, modules=modules)

    elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        class_name = module.__class__.__name__
        lrp_module = conversion_table[class_name].from_torch(module)
        modules.append(lrp_module)
    # maxpool is handled with gradient for the moment

    elif isinstance(module, torch.nn.ReLU): 
        # avoid inplace operations. They might ruin PatternNet pattern
        # computations
        modules.append(torch.nn.ReLU())
    elif isinstance(module, torchvision.models.resnet.Bottleneck):
        # For torchvision Bottleneck
        bottleneck = Bottleneck()
        bottleneck.conv1 = Conv2d.from_torch(module.conv1)
        bottleneck.conv2 = Conv2d.from_torch(module.conv2)
        bottleneck.conv3 = Conv2d.from_torch(module.conv3)
        bottleneck.bn1 = module.bn1
        bottleneck.bn2 = module.bn2
        bottleneck.bn3 = module.bn3
        bottleneck.relu = torch.nn.ReLU()
        if module.downsample is not None:
            bottleneck.downsample = module.downsample
            bottleneck.downsample[0] = Conv2d.from_torch(module.downsample[0])
        modules.append(bottleneck)
    else:
        modules.append(module)

# # # # # Convert torch.models.vggxx to lrp model
def convert_vgg(module, modules=None):
    # First time
    if modules is None: 
        modules = []
        for m in module.children():
            convert_vgg(m, modules=modules)

            # Vgg model has a flatten, which is not represented as a module
            # so this loop doesn't pick it up.
            # This is a hack to make things work.
            if isinstance(m, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)): 
                modules.append(torch.nn.Flatten())

        sequential = Sequential(*modules)
        return sequential

    # Recursion
    if isinstance(module, torch.nn.Sequential): 
        for m in module.children():
            convert_vgg(m, modules=modules)

    elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        class_name = module.__class__.__name__
        lrp_module = conversion_table[class_name].from_torch(module)
        modules.append(lrp_module)
    # maxpool is handled with gradient for the moment

    elif isinstance(module, torch.nn.ReLU): 
        # avoid inplace operations. They might ruin PatternNet pattern
        # computations
        modules.append(torch.nn.ReLU())
    else:
        modules.append(module)

