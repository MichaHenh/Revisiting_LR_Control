# Revisiting Learning Rate Control
The learning rate is one of the most important hyperparameters in deep learning, and how to control it is an active area within both AutoML and deep learning research. 
Approaches for learning rate control span from classic optimization to online scheduling based on gradient statistics. 
This paper compares paradigms to assess the current state of learning rate control. 
We find that methods from multi-fidelity hyperparameter optimization, fixed-hyperparameter schedules, and hyperparameter-free learning often perform very well on selected deep learning tasks but are not reliable across settings. 
This highlights the need for algorithm selection methods in learning rate control, which have been neglected so far by both the AutoML and deep learning communities.
We also observe a trend of hyperparameter optimization approaches becoming less effective as models and tasks grow in complexity, even when combined with multi-fidelity approaches for more expensive model trainings. 
A focus on more relevant test tasks and new promising directions like finetunable methods and meta-learning will enable the AutoML community to significantly strengthen its impact on this crucial factor in deep learning.

## Installation
We recommend installing the dependencies in a conda environemnt.
```
conda create -n lrcontrol python=3.10
conda activate lrcontrol
```
The logistic regression and computer vision experiments require the development version of DACBench.
```
git clone https://github.com/automl/DACBench.git
git checkout development
cd DACBench
pip install .[sgd]
```
Additionally, you need to replace the env_util.py in your dacbench installation folder in the conda environment with this file.
For experiments including SMAC, you need to install it:
```
pip install smac
```

## Minimal Example

## Experiments
All experiments described in this Bachelor's Thesis have a corresponding config file. These can be found [here](ba_parameter_free_dac/configs/). E.g. to execute default AdamW on MNIST for seeds 1,2 and 3, run the following:

```
conda activate YourEnvironment
cd ba_ba_parameter_free_dac
python cli.py --config-name=adamfixed_mnist seed=1,2,3 -m
```

Depending on whether you are running the job locally or on a SLURM cluster or if you want to use GPUs, you might need to choose cluster/local or cluster/cpu in [base.yaml](ba_parameter_free_dac/configs/base.yaml).

To run the meta-training for SMAC Policy you can either execute the [runscripts](ba_parameter_free_dac/runscripts/) or you directly query the python script:

```
conda activate YourEnvironment
cd ba_ba_parameter_free_dac
python smac_policy.py --config-name=smacpolicy seed=2 -m
```