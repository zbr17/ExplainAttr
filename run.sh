# mnist
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset mnist
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset mnist --reduction norm
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset mnist --line_type deletion
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset mnist --line_type insertion

CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset mnist --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset mnist --reduction norm --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset mnist --line_type deletion --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset mnist --line_type insertion --momen 0.5

CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset mnist
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset mnist --reduction norm
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset mnist --line_type deletion
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset mnist --line_type insertion

CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset mnist --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset mnist --reduction norm --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset mnist --line_type deletion --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset mnist --line_type insertion --momen 0.5

# cifar10
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset cifar10
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset cifar10 --reduction norm
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset cifar10 --line_type deletion
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset cifar10 --line_type insertion

CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset cifar10 --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset cifar10 --reduction norm --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset cifar10 --line_type deletion --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset cifar10 --line_type insertion --momen 0.5

CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset cifar10
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset cifar10 --reduction norm
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset cifar10 --line_type deletion
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset cifar10 --line_type insertion

CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset cifar10 --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset cifar10 --reduction norm --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset cifar10 --line_type deletion --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset cifar10 --line_type insertion --momen 0.5

# imagenet
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset imagenet
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset imagenet --reduction norm
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset imagenet --line_type deletion
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset imagenet --line_type insertion

CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset imagenet --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset imagenet --reduction norm --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset imagenet --line_type deletion --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam --dataset imagenet --line_type insertion --momen 0.5

CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset imagenet
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset imagenet --reduction norm
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset imagenet --line_type deletion
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset imagenet --line_type insertion

CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset imagenet --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset imagenet --reduction norm --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset imagenet --line_type deletion --momen 0.5
CUDA_VISIBLE_DEVICES=1 python examples/run_benchmark.py --xai_name sam_naive --dataset imagenet --line_type insertion --momen 0.5
