# mnist
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name <xai-method> --dataset mnist

# cifar10
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name <xai-method> --dataset cifar10

# imagenet
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name <xai-method> --dataset imagenet