import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import AdamW
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from dacbench.runner import run_benchmark
from parameterfree.parameter_free_sgd_benchmark import ParameterFreeSGDBenchmark
from dacbench.agents import StaticAgent
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.benchmarks import SGDBenchmark
from dacbench.logger import Logger
from pathlib import Path
from smac.runhistory.dataclasses import TrialValue

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class DAC(nn.Module):

    def __init__(self, seed) -> None:
        # Get benchmark env
        bench = SGDBenchmark()
        self.env = bench.get_benchmark(seed=seed)
        
        self.env = PerformanceTrackingWrapper(self.env)
        super().__init__()

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        lr = Float("lr", (0, 0.005), default=1e-3)
        cs.add_hyperparameters([lr])

        return cs

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def train_smac(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        lr = config["lr"]

        run_benchmark(self.env, StaticAgent(self.env, lr), 1)

        performance = self.env.get_performance()

        print(f"Current Performance: {performance}")
        print(f"Current Performance: {performance[0][-1]}")

        return performance[0][-1]
    

model = DAC(119).to(device)

scenario = Scenario(model.configspace, deterministic=True, n_trials=50)

smac = HPOFacade(
        scenario,
        model.train_smac,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )

incumbent = smac.optimize()



print(f"Incumbent: {incumbent}")

# Get cost of default configuration
default_cost = smac.validate(model.configspace.get_default_configuration())
print(f"Default cost: {default_cost}")

# Let's calculate the cost of the incumbent
incumbent_cost = smac.validate(incumbent)
print(f"Incumbent cost: {incumbent_cost}")


### Test Model using DACBench


def setup_env(seed):
    # Get benchmark env
    bench = SGDBenchmark()
    env = bench.get_benchmark(seed=seed)
    
    # Make logger to write results to file
    logger = Logger(experiment_name=f"smac_best_fixed_s{seed}", output_path=Path("results"))
    perf_logger = logger.add_module(PerformanceTrackingWrapper)
    
    env = PerformanceTrackingWrapper(env, logger=perf_logger)
    logger.set_env(env)
    logger.set_additional_info(seed=seed)
    
    return env, logger

for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    env, logger = setup_env(seed)
    
    # This could be any optimization or learning method
    agent = StaticAgent(env, incumbent["lr"])
    run_benchmark(env, agent, num_episodes=30, logger=logger)
