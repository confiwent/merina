# Artifact Appendix for MERINA 
Code for the paper "Improving Generalization for Neural Adaptive Video Streaming via Meta Reinforcement Learning" - Nuowen Kan et al. (ACM MM22)

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
The main training loop for _MERINA_ can be found in ```main.py```, the trained models are in ```models/``` and the corresponding meta RL algorithms in ```algos/```. Besides, ```envs/``` includes emulator codes which simulate the environment of ABR virtual player. Some of baseline aglorithms are located in ```baselines/```.

There's quite a bit of documentation in the respective scripts so have a look there for details. 

__Improvements to the codes and methods (including a journal version) are currently underway  and will be finished in a few months. I will update them all in this repository.__

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