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
The logistic regression and computer vision experiments require the following packages:
```
pip install dacbench torch torchvision hydra-core timm download smac ioh rliable
```
To make DACBench compatible with additional datasets, you need to copy the folders [envs](ba_parameter_free_dac/dacbench_custom/envs/) and [instance_sets](ba_parameter_free_dac/dacbench_custom/instance_sets/) into your dacbench installation folder. The path will look something like this "/.conda/envs/lrcontrol/lib/python3.10/site-packages/dacbench/".

## Minimal Example
The smallest example we provide is running Adam on the LIBSVM dataset iris:
```
cd ba_parameter_free_dac
python cli.py --config-name=adamfixed_libsvm dacbench_sgd_config.dataset_name=iris seed=1,2,3 -m
```

## Experiments
All experiments described in this paper have a corresponding config file. These can be found [here](ba_parameter_free_dac/configs/). The runs are executed either via [cli.py](ba_parameter_free_dac/cli.py) (for computer vision and logistic regression) or [roberta_experiment.py](ba_parameter_free_dac/roberta_experiment.py) (for nlp).E.g. to execute D-Adaptation on CIFAR-10 for seeds 1,2 and 3, run the following:
```
python cli.py --config-name=dadaptation_cifar10 seed=1,2,3 -m
```
The natural language processing experiments can be started like this:
```
python roberta_experiment.py --config-name=dadaptation_bookwiki_roberta seed=1 -m
```
Depending on whether you are running the job locally or on a SLURM cluster or if you want to use GPUs, you might need to choose cluster/local or cluster/cpu in [base.yaml](ba_parameter_free_dac/configs/base.yaml) and adjust wall times and resource allocations to fit your environment.
