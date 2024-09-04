# Comparing Parameter-Free Optimization and Black-Box Optmization for Dynamic Algorithm Configuration
Setting hyperparameters of gradient-based optimization algorithms such as SGD and Adam is crucial to achieve the best performance. Hyperparameter Optimization (HPO) and Dynamic Algorithm Configuration (DAC) succeed at this task. However, this comes at a substantial resource cost. Therefore, *parameter-free* optimizers (PFO) - aiming to eliminate hyperparameters while still ensuring adaptivity and close-to-optimal performance - pose an attractive alternative. In this thesis, we compare two DAC approaches with two PFO approaches on standard benchmark data sets MNIST, FashionMNIST, CIFAR-10, and CIFAR-100. Furthermore, we show that in this setting, DAC generally outperforms PFO by up to 7% in test accuracy.

## Experiments
All experiments described in this Bachelor's Thesis have a corresponding config file. These can be found [here](ba_parameter_free_dac/configs/). E.g. to execute default AdamW on MNIST for seeds 1,2 and 3, run the following:

```
cd ba_ba_parameter_free_dac
python cli.py --config-name=adamfixed_mnist seed=1,2,3 -m
```

Depending on whether you are running the job locally or on a SLURM cluster or if you want to use GPUs, you might need to choose cluster/local or cluster/cpu in [base.yaml](ba_parameter_free_dac/configs/base.yaml).