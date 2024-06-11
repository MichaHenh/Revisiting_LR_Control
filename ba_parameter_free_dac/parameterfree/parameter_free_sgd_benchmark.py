from dacbench.benchmarks.sgd_benchmark import SGDBenchmark, SGDEnv, SGD_DEFAULTS
from dacbench.abstract_benchmark import objdict
from torch.optim.optimizer import Optimizer
import torch
import numpy as np
from dacbench.envs.env_utils import utils

class ParameterFreeSGDBenchmark(SGDBenchmark):

    def __init__(self, optimizer, config_path=None, config=None):
        super().__init__(config_path, config)
        self.optimizer_type = optimizer

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

        env = ParameterFreeSGDEnv(self.optimizer_type, self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env
    
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
        return ParameterFreeSGDEnv(self.optimizer_type, self.config)


class ParameterFreeSGDEnv(SGDEnv):

    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_type = optimizer

    def reset(self, seed=None, options=None):
        """Initialize the neural network, data loaders, etc. for given/random next task.
        Also perform a single forward/backward pass,
        not yet updating the neural network parameters.
        """
        if options is None:
            options = {}
        super().reset_(seed)

        # Use generator
        rng = np.random.RandomState(self.initial_seed)
        if self.use_generator:
            (
                self.model,
                self.optimizer_params,
                self.batch_size,
                self.crash_penalty,
            ) = utils.random_instance(rng, self.datasets)
        else:
            # Load model from config file
            self.model = utils.create_model(
                self.config.get("layer_specification"), len(self.datasets[0].classes)
            )

        self.learning_rate = None
        # self.optimizer_type = torch.optim.AdamW
        self.info = {}
        self._done = False

        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = self.optimizer_type(params=self.model.parameters())
        self.loss = 0
        self.test_losses = None

        self.validation_loss = 0
        self.min_validation_loss = None

        return self.get_state(self), {}
