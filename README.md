# Continual Neural Dirichlet Process Mixture
Official PyTorch implementation of ICLR 2020 paper: *A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning*.

[**Paper**](https://openreview.net/forum?id=SJxSOJStPr)

![CN-DPM](./images/cndpm.png)

## System Requirements
- Python >= 3.6.1
- CUDA >= 9.0 supported GPU with at least 10GB memory


## Installation
```bash
$ pip install -r requirements.txt
```

## Usage
```bash
$ python main.py --help
usage: main.py [-h] [--config CONFIG] [--episode EPISODE] [--log-dir LOG_DIR]
            [--resume-ckpt RESUME_CKPT] [--override OVERRIDE]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
  --episode EPISODE, -e EPISODE
  --log-dir LOG_DIR, -l LOG_DIR
  --override OVERRIDE
```


## Composing Continual Learning Episodes
We provide quick and easy solution to compose a continual learning scenario.
You can configure a scenario by writing a YAML file.
Here is an example of Split-CIFAR10 where each stage is repeated for 10 epochs: 
```yaml
- subsets: [['cifar10', 0], ['cifar10', 1]]
  epochs: 10
- subsets: [['cifar10', 2], ['cifar10', 3]]
  epochs: 10
- subsets: [['cifar10', 4], ['cifar10', 5]]
  epochs: 10
- subsets: [['cifar10', 6], ['cifar10', 7]]
  epochs: 10
- subsets: [['cifar10', 8], ['cifar10', 9]]
  epochs: 10
```
Basic rules:
- Each scenario consists of a list of stages.
- Each stage defines a list of subsets.
- A subset is a two-element list `[dataset_name, subset_name]`. By default, each class is defined as a subset with the class number as its name.
- Each stage may optionally define one of `epochs`, `steps`, and `samples` to set the length of the stage. Otherwise, the default length is set to 1 epoch.

The main logic is implemented in the `DataScheduler` in `data.py`.


## Reproducing Experiments
Run below commands to reproduce our experimental results. You can check summaries from Tensorboard.

### 1. MNIST Generation
#### iid Offline
```bash
$ python main.py \
    --config configs/mnist_gen-iid_offline.yaml \
    --episode episodes/mnist-iid-100epochs.yaml \
    --log-dir log/mnist_gen-iid_offline
```
#### iid Online
```bash
$ python main.py \
    --config configs/mnist_gen-iid_online.yaml \
    --episode episodes/mnist-iid-online.yaml \
    --log-dir log/mnist_gen-iid_online
```
#### Finetune
```bash
$ python main.py \
    --config configs/mnist_gen-iid_online.yaml \
    --episode episodes/mnist-split-online.yaml \
    --log-dir log/mnist_gen-finetune
```
#### Reservoir
```bash
$ python main.py \
    --config configs/mnist_gen-reservoir.yaml \
    --episode episodes/mnist-split-online.yaml \
    --log-dir log/mnist_gen-reservoir
```
#### CN-DPM
```bash
$ python main.py \
    --config configs/mnist_gen-cndpm.yaml \
    --episode episodes/mnist-split-online.yaml \
    --log-dir log/mnist_gen-cndpm
```

### 2. MNIST Classification
#### iid Offline
```bash
$ python main.py \
    --config configs/mnist-iid_offline.yaml \
    --episode episodes/mnist-iid-100epochs.yaml \
    --log-dir log/mnist-iid_offline
```
#### iid Online
```bash
$ python main.py \
    --config configs/mnist-iid_online.yaml \
    --episode episodes/mnist-iid-online.yaml \
    --log-dir log/mnist-iid_online
```
#### Finetune
```bash
$ python main.py \
    --config configs/mnist-iid_online.yaml \
    --episode episodes/mnist-split-online.yaml \
    --log-dir log/mnist-finetune
```
#### Reservoir
```bash
$ python main.py \
    --config configs/mnist-reservoir.yaml \
    --episode episodes/mnist-split-online.yaml \
    --log-dir log/mnist-reservoir
```
#### CN-DPM
```bash
$ python main.py \
    --config configs/mnist-cndpm.yaml \
    --episode episodes/mnist-split-online.yaml \
    --log-dir log/mnist-cndpm
```

### 3. MNIST-SVHN Classification
#### iid Offline
```bash
$ python main.py \
    --config configs/mnist_svhn-iid_offline.yaml \
    --episode episodes/mnist_svhn-iid-10epochs.yaml \
    --log-dir log/mnist_svhn-iid_offline
```
#### iid Online
```bash
$ python main.py \
    --config configs/mnist_svhn-iid_online.yaml \
    --episode episodes/mnist_svhn-iid-online.yaml \
    --log-dir log/mnist_svhn-iid_online
```
#### Finetune
```bash
$ python main.py \
    --config configs/mnist_svhn-iid_online.yaml \
    --episode episodes/mnist_svhn-online.yaml \
    --log-dir log/mnist_svhn-finetune
```
#### Reservoir
```bash
$ python main.py \
    --config configs/mnist_svhn-reservoir.yaml \
    --episode episodes/mnist_svhn-online.yaml \
    --log-dir log/mnist_svhn-reservoir
```
#### CN-DPM
```bash
$ python main.py \
    --config configs/mnist_svhn-cndpm.yaml \
    --episode episodes/mnist_svhn-online.yaml \
    --log-dir log/mnist_svhn-cndpm
```

### 4. CIFAR10 Classification
#### iid Offline
```bash
$ python main.py \
    --config configs/cifar10-iid_offline.yaml \
    --episode episodes/cifar10-iid-100epochs.yaml \
    --log-dir log/cifar10-iid_offline
```
#### iid Online
```bash
$ python main.py \
    --config configs/cifar10-iid_online.yaml \
    --episode episodes/cifar10-iid-online.yaml \
    --log-dir log/cifar10-iid_online
```
#### Finetune
```bash
$ python main.py \
    --config configs/cifar10-iid_online.yaml \
    --episode episodes/cifar10-split-online.yaml \
    --log-dir log/cifar10-finetune
```
#### Reservoir
```bash
$ python main.py \
    --config configs/cifar10-reservoir.yaml \
    --episode episodes/cifar10-split-online.yaml \
    --log-dir log/cifar10-reservoir
```
#### CN-DPM
```bash
$ python main.py \
    --config configs/cifar10-cndpm.yaml \
    --episode episodes/cifar10-split-online.yaml \
    --log-dir log/cifar10-cndpm
```
#### CN-DPM (0.2 Epoch)
```bash
$ python main.py \
    --config configs/cifar10-cndpm.yaml \
    --episode episodes/cifar10-split-0.2epoch.yaml \
    --log-dir log/cifar10-cndpm-0.2epoch
```
#### CN-DPM (10 Epochs)
```bash
$ python main.py \
    --config configs/cifar10-cndpm.yaml \
    --episode episodes/cifar10-split-10epochs.yaml \
    --log-dir log/cifar10-cndpm-10epoch
```

### 5. CIFAR100 Classification
#### iid Offline
```bash
$ python main.py \
    --config configs/cifar100-iid_offline.yaml \
    --episode episodes/cifar100-iid-100epochs.yaml \
    --log-dir log/cifar100-iid_offline
```
#### iid Online
```bash
$ python main.py \
    --config configs/cifar100-iid_online.yaml \
    --episode episodes/cifar100-iid-online.yaml \
    --log-dir log/cifar100-iid_online
```
#### Finetune
```bash
$ python main.py \
    --config configs/cifar100-iid_online.yaml \
    --episode episodes/cifar100-split-online.yaml \
    --log-dir log/cifar100-finetune
```
#### Reservoir
```bash
$ python main.py \
    --config configs/reservoir-resnet_classifier-cifar100.yaml \
    --episode episodes/cifar100-split-online.yaml \
    --log-dir log/cifar100-reservoir
```
#### CN-DPM
```bash
$ python main.py \
    --config configs/cifar100-cndpm.yaml \
    --episode episodes/cifar100-split-online.yaml \
    --log-dir log/cifar100-cndpm
```
