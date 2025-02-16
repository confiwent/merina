<h1 align="center"> [🔥 ACM MM22 Top Rated Paper] Improving Generalization for Neural Adaptive Video Streaming via Meta Reinforcement Learning </h1>

<p align="center">
    <a href="https://dl.acm.org/doi/abs/10.1145/3503161.3548331"><img src="https://img.shields.io/badge/MM22-fp2586-blue.svg" alt="Paper"></a>
    <a href="https://github.com/confiwent/merina"><img src="https://img.shields.io/badge/Github-MERINA-brightgreen?logo=github" alt="Github"></a>
</p>

by Nuowen Kan, Yuankun Jiang, Chenglin Li, Wenrui Dai, Junni Zou, and Hongkai Xiong at Shanghai Jiao Tong University

This repository is the official Pytorch implementation of MERINA

## Abstract
This document illustrates how to obtain the results shown in the paper "Improving Generalization for Neural Adaptive Video Streaming via Meta Reinforcement Learning".

## The environment setup

_Anaconda is suggested to be installed to manage the test environments._

### Prerequisites
- Linux or macOS
- Python >=3.6
- Pytorch >= 1.6.0
- numpy, pandas
- tqdm
- seaborn
- matplotlib
- CPU or NVIDIA GPU + CUDA CuDNN

Install PyTorch. Note that the command of PyTorch intallation depends on the actual compute platform of your own computer, and you can choose appropriate version following the [guide page](https://pytorch.org/get-started/locally/). For example, if you have intalled `CUDA 10.2`, you can intall PyTorch with the latest version by running this Command:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

__Or__ you can create a specific environment (_many redundant dependencies will be installed_) just via
```
conda env create -f torch.yaml
```

## Overview
The main training loop for _MERINA_ can be found in ```main.py```, the trained models are in ```models/``` and the corresponding meta RL algorithms in ```algos/```. Besides, ```envs/``` includes emulator codes which simulate the environment of ABR virtual player. Some of baseline algorithms are located in ```baselines/```.

There's quite a bit of documentation in the respective scripts so have a look there for details. 

__Improvements to the codes and methods (including a journal version which is undergoing a major revision) are provided in the repository below:__
<p align="left">
    <a href="https://github.com/confiwent/merina-plus"><img src="https://img.shields.io/badge/Github-MERINA_Plus-brightgreen?logo=github" alt="Github"></a>
</p>

## Usage
Type ```python main.py -h``` for the instruction in the terminal, or read the description of ArgumentParser part.

> For example, you could choose work mode by selecting the argument ```--test``` [evaluate the model], ```--adp``` [run the meta adaptation procedure] or use the default setting [run the meta training procedure]; 
> 
> The default QoE metric is the linear form and you can change it to logrithmic form by add the argument ```--log```;
>
> The network throughput dataset must be chosen in the __test__ mode. The datasets shown in the paper are available here and you can select them following the mappings below:
> 
> ```[--tf FCC traces] [--tfh FCC and HSDPA traces] [--t3g HSDPA traces] [--to Oboe traces] [--tp Puffer-Oct.17-21 traces] [--tp2 Puffer-Feb.18-22 traces]```
>
> Also, you can rename the lable of the results by ```[--name "yourname"]```

## Create the Results folders
``` 
    |--models
    |--Results  # !!
       |--sim
       |--test
          |--lin
          |--log
    |--utils
       |--log_results # !!
    |--main.py
```

## Load the network throughput traces
The public bandwidth traces are stored in this [repository](https://github.com/confiwent/Real-world-bandwidth-traces). Download and put them in the directory `./envs/traces/`.

## Runnig an experiment
To evalute _MERINA_ on the in-distribution throughput traces with the $QoE_{log}$ metric from the paper, run

- FCC traces 
```
python main.py --test --tf --log
``` 

- or HSDPA traces
```
python main.py --test --t3g --log
```

Plot a example result
```
cd utils
python plt_v2.py --log --merina --bola --mpc
```

## Try to train a model from scratch

To train a model using the FCC and HSDPA training dataset with the $QoE_{log}$ metric from the paper, just run
```
python main.py --log
```
The exploration trajectories will be shown in ```./Results/sim/merina/log_record``` and the __valid results__ are in ```./Results/sim/merina/log_test```; In addition, you can monitor the training process using tensorboard, run
```
tensorboard --logdir=./Results/sim
```

Then, wait patiently and mannually interrupt the training (```Ctrl + C``` in the terminal) when the __valid results__ converges. Cross your fingers!!!

## Comments

- The script ```imrl_light.py``` is a variant that employs lightweight neural networks to build the VAE and policy network. Because it is an unfinished version, some problems may arise if you use it to train the models.