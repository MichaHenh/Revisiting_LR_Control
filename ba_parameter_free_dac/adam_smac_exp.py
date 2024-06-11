import torch
import json
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from dacbench.runner import run_benchmark
from dacbench.wrappers import PolicyProgressWrapper
from dacbench_custom.custom_tracking_wrapper import CustomTrackingWrapper
from dacbench.benchmarks import SGDBenchmark
from dacbench.logger import Logger
from pathlib import Path
from dacbench.abstract_agent import AbstractDACBenchAgent
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

class SMACLRAgent(AbstractDACBenchAgent):
    """Agent for Learning Rate in SGD Benchmark using SMAC"""

    def __init__(self, env, configspace, n_trials):
        """Initialize the Agent."""
        self.scenario = Scenario(configspace, deterministic=True, n_trials=n_trials)

        def dummy_train(self, config: Configuration, seed: int = 0) -> float:
            pass

        self.smac = HPOFacade(
            self.scenario,
            dummy_train,  # We pass the target function here
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
            )
        super().__init__(env)

    def act(self, state=None, reward=None):
        """Returns the next action."""
        print("SMAC act")
        self.current_info = self.smac.ask()
        assert self.current_info.seed is not None

        return self.current_info.config["lr"]

    def train(self, state=None, reward=None):  # noqa: D102
        value = TrialValue(cost=reward, time=0.5)
        self.smac.tell(self.current_info, value=value)

    def end_episode(self, state=None, reward=None):  # noqa: D102
        pass



#print(f"Incumbent: {incumbent}")

# Get cost of default configuration
#default_cost = smac.validate(model.configspace.get_default_configuration())
#print(f"Default cost: {default_cost}")

# Let's calculate the cost of the incumbent
#incumbent_cost = smac.validate(incumbent)
#print(f"Incumbent cost: {incumbent_cost}")


### Test Model using DACBench


def setup_env(seed):
    # Get benchmark env
    bench = SGDBenchmark()
    env = bench.get_benchmark(seed=seed)
    
    # Make logger to write results to file
    logger = Logger(experiment_name=f"smac_s{seed}", output_path=Path("results"))
    perf_logger = logger.add_module(CustomTrackingWrapper)
    logger.add_module(PolicyProgressWrapper)
    
    env = CustomTrackingWrapper(env, logger=perf_logger)
    def dummy(d):
        return 0
    env = PolicyProgressWrapper(env, dummy)
    logger.set_env(env)
    logger.set_additional_info(seed=seed)
    
    return env, logger

for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    env, logger = setup_env(seed)
    
    cs = ConfigurationSpace(seed=seed)
    lr = Float("lr", (0, 0.01), default=1e-3)
    cs.add_hyperparameters([lr])

    # This could be any optimization or learning method
    agent = SMACLRAgent(env, cs, 3000)
    run_benchmark(env, agent, num_episodes=30, logger=logger)
    with open('results/' + logger.experiment_name + "/Policy.jsonl", 'w') as file:
        json.dump(env.policy_progress, file)