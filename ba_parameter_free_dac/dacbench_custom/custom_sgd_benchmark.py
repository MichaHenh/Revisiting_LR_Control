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
from dacbench.benchmarks.sgd_benchmark import SGDBenchmark, SGD_DEFAULTS
from dacbench_custom.custom_sgd import CustomSGDEnv
import dacbench

class CustomSGDBenchmark(SGDBenchmark):
    """Benchmark with default configuration & relevant functions for SGD."""

    def __init__(self, optimizer_type, config_path=None, config=None):
        """Initialize SGD Benchmark.

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        #if "instance_set_path" not in config:
        #    config["instance_set_path"] = str(Path(dacbench.envs.__file__).resolve().parent / SGD_DEFAULTS["instance_set_path"])
        super(CustomSGDBenchmark, self).__init__(config_path, config)
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
