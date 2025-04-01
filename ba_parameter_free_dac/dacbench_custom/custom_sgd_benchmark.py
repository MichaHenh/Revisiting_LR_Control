"""Custom Benchmark for SGD."""
from __future__ import annotations

import csv
from pathlib import Path

import ConfigSpace as CS  # noqa: N817
import dacbench.envs
import numpy as np
from gymnasium import spaces
from torch import nn

from dacbench.abstract_benchmark import objdict
from dacbench.benchmarks.sgd_benchmark import SGDBenchmark
from dacbench_custom.custom_sgd import CustomSGDEnv
from dacbench.envs.env_utils import sgd_utils
import dacbench
from dacbench.abstract_benchmark import AbstractBenchmark

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
LR = CS.Float(name="learning_rate", bounds=(0.0, 0.05))
# Value used for momentum like adaptation, as adam optimizer has no real momentum;
# "beta1" is changed
MOMENTUM = CS.Float(
    name="momentum", bounds=(0.0, 1.0)
)  # ! Only used, when "use_momentum" var in config true
DEFAULT_CFG_SPACE.add_hyperparameter(LR)
DEFAULT_CFG_SPACE.add_hyperparameter(MOMENTUM)


def __default_loss_function(**kwargs):
    return nn.NLLLoss(reduction="none", **kwargs)


INFO = {
    "identifier": "LR",
    "name": "Learning Rate Adaption for Neural Networks",
    "reward": "Negative Log Differential Validation Loss",
    "state_description": [
        "Step",
        "Loss",
        "Validation Loss",
        "Crashed",
    ],
    "action_description": ["Learning Rate", "Momentum"],
}


SGD_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "step": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "loss": spaces.Box(0, np.inf, shape=(1,)),
                "validationLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "crashed": spaces.Discrete(1),
            }
        ],
        "reward_range": [-(10**9), (10**9)],
        "device": "cpu",
        "model_from_dataset": False,  # If true, generates:
        # random model, optimizer_params, batch_size, crash_penalty
        "layer_specification": [
            (
                sgd_utils.LayerType.CONV2D,
                {"in_channels": 1, "out_channels": 32, "kernel_size": 3},
            ),
            (sgd_utils.LayerType.RELU, {}),
            (
                sgd_utils.LayerType.CONV2D,
                {"in_channels": 32, "out_channels": 64, "kernel_size": 3},
            ),
            (sgd_utils.LayerType.RELU, {}),
            (sgd_utils.LayerType.POOLING, {"kernel_size": 2}),
            (sgd_utils.LayerType.DROPOUT, {"p": 0.25}),
            (sgd_utils.LayerType.FLATTEN, {"start_dim": 1}),
            (
                sgd_utils.LayerType.LINEAR,
                {"in_features": 9216, "out_features": 128},
            ),
            (sgd_utils.LayerType.RELU, {}),
            (sgd_utils.LayerType.DROPOUT, {"p": 0.25}),
            (sgd_utils.LayerType.LINEAR, {"in_features": 128, "out_features": 10}),
            (sgd_utils.LayerType.LOGSOFTMAX, {"dim": 1}),
        ],
        "torch_hub_model": (False, False, False),
        "optimizer_params": {
            "weight_decay": 0,
            "eps": 1e-8,
            "betas": (0.9, 0.99),
        },
        "cutoff": 1e2,
        "loss_function": __default_loss_function,
        "loss_function_kwargs": {},
        "training_batch_size": 64,
        "fraction_of_dataset": 0.6,
        "train_validation_ratio": 0.8,  # If set to None, random value is used
        "dataset_name": "MNIST",  # If set to None, random data set is chosen;
        # else specific set can be set: e.g. "MNIST"
        # "reward_function":,    # Can be set, to replace the default function
        # "state_method":,       # Can be set, to replace the default function
        "use_momentum": False,
        "seed": 0,
        "crash_penalty": -100.0,
        "multi_agent": False,
        "instance_set_path": "../instance_sets/sgd/sgd_train_100instances.csv",
        "benchmark_info": INFO,
        "epoch_mode": True,
    }
)

class CustomSGDBenchmark(SGDBenchmark):
    r"""Customizable SGD Benchmark with default configuration & relevant functions for SGD. Main differences to SGDBenchmark is a specifiable optimizer_type."""

    def __init__(self, optimizer_type, config_path=None, config=None):
        """Initialize SGD Benchmark.

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super().__init__(config_path, config)
        if not self.config:
            self.config = objdict(SGD_DEFAULTS.copy())

        for key in SGD_DEFAULTS:
            if key not in self.config:
                self.config[key] = SGD_DEFAULTS[key]
        self.optimizer_type = optimizer_type

    def get_environment(self):
        """Return SGDEnv env with current configuration.

        Returns:
        -------
        SGDEnv
            SGD environment
        """
        if "instance_set" not in self.config:
            self.read_instance_set()

        # Read test set if path is specified
        if "test_set" not in self.config and "test_set_path" in self.config:
            self.read_instance_set(test=True)

        env = CustomSGDEnv(self.config, self.optimizer_type)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self, test=False):
        """Read path of instances from config into list."""
        if test:
            path = Path(dacbench.envs.__file__).resolve().parent / self.config.test_set_path
            keyword = "test_set"
        else:
            #path = Path(dacbench.envs.__file__).resolve().parent / self.config.instance_set_path
            path = Path(dacbench.envs.__file__).resolve().parent /'../instance_sets/sgd/sgd_train_100instances.csv' 
            keyword = "instance_set"
        self.config[keyword] = {}
        with open(path) as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for row in reader:
                if "_" in row["dataset"]:
                    dataset_info = row["dataset"].split("_")
                    dataset_name = dataset_info[0]
                    dataset_size = int(dataset_info[1])
                else:
                    dataset_name = row["dataset"]
                    dataset_size = None
                instance = [
                    dataset_name,
                    int(row["seed"]),
                    row["architecture"],
                    int(row["steps"]),
                    dataset_size,
                ]
                self.config[keyword][int(row["ID"])] = instance

    def get_benchmark(self, instance_set_path=None, seed=0):
        """Get benchmark from the LTO paper.

        Parameters
        -------
        seed : int
            Environment seed

        Returns:
        -------
        env : SGDEnv
            SGD environment
        """
        self.config = objdict(SGD_DEFAULTS.copy())
        if instance_set_path is not None:
            self.config["instance_set_path"] = instance_set_path
        self.config.seed = seed
        self.read_instance_set()
        return CustomSGDEnv(self.config, self.optimizer_type)
