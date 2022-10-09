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
- CPU or NVIDIA GPU + CUDA CuDNN

The _MERINA_ should be tested with python 3.6, pytorch 1.6.0, numpy, matplotlib, and pandas.

- Install PyTorch. Note that the command of PyTorch intallation depends on the actual compute platform of your own computer, and you can choose appropriate version following the [guide page](https://pytorch.org/get-started/locally/). For example, if you have intalled `CUDA 10.2`, you can intall PyTorch with the latest version by running this Command:

    ```
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    ```

## Overview
The main training loop for _MERINA_ can be found in ```main.py```, the trained models are in ```models/``` and the corresponding meta RL algorithms in ```algos/```. Besides, ```envs/``` includes emulator codes which simulate the environment of ABR virtual player. Some of baseline aglorithms are located in ```baselines/```.

There's quite a bit of documentation in the respective scripts so have a look there for details. 

__Improvements to the codes and methods (including a journal version) are currently underway  and will be finished in a few months. I will update them all in this repository.__

## Usage
Type ```python main.py -h``` for the instruction in the terminal, or read the description of ArgumentParser part.

> For example, you could choose the work mode by the argument ```--test``` [evaluate the model], ```--adp``` [run the meta adaptation procedure] or use the default setting [run the meta training procedure]; 
> 
> The default QoE metric is the linear form and you can change it to logrithmic form by add the argument ```--log```;
>
> The network throughput dataset must be chosen in the __test__ mode. The datasets shown in the paper are available here and you can select them following the mappings below:
> 
> ```[--tf FCC traces] [--tfh FCC and HSDPA traces] [--t3g HSDPA traces] [--to Oboe traces] [--tp Puffer-Oct.17-21 traces] [--tp2 Puffer-Feb.18-22 traces]```

## Runnig an experiment



### Results of Fig.2
To replicate the results in Fig.2, you should follow the steps below:
1. Create two results folders

    ```
    mkdir ./results_lin
    mkdir ./results_lin/cb_fcc ./results_lin/cb_HSDPA
    ```
2. Generate the results of _BayesMPC_ and RobustMPC in Fig.2(a) and (b).

    ```python
    conda activate bayes
    ## for fig.2(a)
    python bbp_mpc_v3.py --cb --HSDPA
    python mpc_v2.py --cb --HSDPA
    ## for fig.2(b)
    python bbp_mpc_v3.py --cb --FCC
    python mpc_v2.py --cb --FCC
    ```
3. Plot the results in Fig.2(a).

    ```python
    ## for fig.2(a)
    python plot_results_fig2.py --a
    ## for fig.2(b)
    python plot_results_fig2.py --b
    ```

<!-- ![fig2a](./pic/random_traces_prediction_norway.pdf)
![fig2b](./pic/random_traces_prediction_fcc.pdf) -->

### Results of Figs. 3, 4
There are two different QoE metrics: $QoE_{lin}$ and $QoE_{log}$ and two different throughput datasets: FCC and HSDPA. For comparison, all $6$ baseline algorithms should be tested in these settings and the results shown in Figs.3, 4 in Section.3 of the paper can be replicated by following commands:
1. Create the results folders for different QoE metrics and different throughput datasets.

    ```python
    ## create folder for QoE metric QoE_lin
    mkdir ./results_lin
    ## create folder for different datasets with the QoE_lin metric
    mkdir ./results_lin/fcc ./results_lin/HSDPA
    ## same for QoE metric QoE_log
    mkdir ./results_log
    mkdir ./results_log/fcc ./results_log/HSDPA
    ```

2. Test _BayesMPC_ for different QoE metrics and for different throughput datasets.

    ```python
    conda activate bayes
    python bbp_mpc_v3.py --lin --HSDPA
    python bbp_mpc_v3.py --lin --FCC
    python bbp_mpc_v3.py --log --HSDPA
    python bbp_mpc_v3.py --log --FCC
    ## deactivate the virtual environment
    conda deactivate
    ```

3. Test Pensieve for different QoE metrics and for different throughput datasets.

    ```python
    conda activate pensieve
    python rl_no_training.py --lin --HSDPA
    python rl_no_training.py --lin --FCC
    python rl_no_training.py --log --HSDPA
    python rl_no_training.py --log --FCC
    ## deactivate the virtual environment
    conda deactivate
    ```

4. Test other baseline algorithms in ```pensieve``` virtual environment. Note that the following commands are illustrated for testing RobustMPC, and other baseline algorithms can be tested by just replacing the file name ```mpc_v2.py``` with the corresponding file name (```Bola_v1.py```, ```bb.py```, ```rb.py```).

    ```python
    conda activate pensieve
    python mpc_v2.py --lin --HSDPA
    python mpc_v2.py --lin --FCC
    python mpc_v2.py --log --HSDPA
    python mpc_v2.py --log --FCC
    ## deactivate the virtual environment
    conda deactivate
    ```

5. __Plot the results__: The results in Figs. 3 and 4 for different QoE metrics and different datasets can be plotted by the following commands:

    ```python
    conda activate pensieve
    # for results tested with QoE_lin and HSDPA dataset in Fig. 3 and 4(a)
    python plot_results_fig34.py --lin --HSDPA
    # for results tested with QoE_log and HSDPA dataset in Fig. 3 and 4(b)
    python plot_results_fig34.py --log --HSDPA
    # for results tested with QoE_lin and FCC dataset in Fig. 3 and 4(c)
    python plot_results_fig34.py --lin --FCC
    # for results tested with QoE_log and FCC dataset in Fig. 3 and 4(d)
    python plot_results_fig34.py --log --FCC
    ## deactivate the virtual environment
    conda deactivate
    ```

### Results of Fig.6
The procedure for plotting Figure 6 is similar to the procedure for Figure 3, 4. The only difference is test throughput dataset for baseline algorithms. The results of Fig.6 are tested with Oboe dataset.

- Test _BayesMPC_ with Oboe dataset.

    ```python
    conda activate bayes
    python bbp_mpc_v3.py --lin --Oboe
    python bbp_mpc_v3.py --log --Oboe
    ## deactivate the virtual environment
    conda deactivate
    ```

- Test Pensieve with Oboe dataset. 

    ```python
    conda activate pensieve
    python rl_no_training.py --lin --Oboe
    python rl_no_training.py --log --Oboe
    ## deactivate the virtual environment
    conda deactivate
    ```

- Test other algorithms with Oboe dataset. Note that the following commands are illustrated for testing RobustMPC, and other baseline algorithms can be tested by just replacing the file name ```mpc_v2.py``` with the corresponding file name (```Bola_v1.py```, ```bb.py```, ```rb.py```).

    ```python
    conda activate pensieve
    python mpc_v2.py --lin --Oboe
    python mpc_v2.py --log --Oboe
    ## deactivate the virtual environment
    conda deactivate
    ```

- __Plot the results__: The results in Fig.6 for different QoE metrics with Oboe datasets can be plotted by the following commands:

    ```python
    conda activate pensieve
    # for results tested with QoE_lin and Oboe dataset in Fig. 6(a)
    python plot_results_fig34.py --lin --Oboe
    # for results tested with QoE_log and Oboe dataset in Fig. 6(b)
    python plot_results_fig34.py --log --Oboe
    ## deactivate the virtual environment
    conda deactivate
    ```
