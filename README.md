# Benchmark for Explainable Attributions

## Preparation

### Dataset

#### MNIST

- Download from [here](http://yann.lecun.com/exdb/mnist/)

#### CIFAR-10

- Download from [here](https://www.cs.toronto.edu/~kriz/cifar.html)

#### ImageNet

- Download from [here](https://www.image-net.org/)

Organize ImageNet as follows:

```
- dataset
    |- train
    |   |- class1
    |   |   |- image1
    |   |   |- ...
    |   |- ...
    |- test
        |- class1
        |   |- image1
        |   |- ...
        |- ...
```

### Pre-trained models

We can download pre-trained models for visualization from [here](https://ufile.io/utqk0zrc), and put the files in the [log folder](log).

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Device 

We tested our code on a linux machine with an Nvidia RTX 3090 GPU card. We recommend using a GPU card with a memory > 8GB.

## Visualization of Attributions

To visulize the attributions, please run the command as follows:
```bash
python examples/run_benchmark.py --xai_name <...> --dataset <...>
```

Avaliable argument options:
| Argument | Option |
| - | - |
| xai_name | sam / sam_naive / ig / cam / gradcam / gradcamplus / rise / sg / lrp / deeplift / lime / xrai / blurig / guidedig |
| dataset | mnist / cifar10 / imagenet |

## Deletion / Insertion

To test our SAMP with Deletion/Insertion metrics, please run the command as follows:
```bash
bash run.sh
```

## Reference Repos

- [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
- [TorchLRP](https://github.com/fhvilshoj/TorchLRP)
- [captum](https://github.com/pytorch/captum)
- [lime](https://github.com/marcotcr/lime)
- [saliency](https://github.com/PAIR-code/saliency)