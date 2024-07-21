import math
from dacbench.abstract_agent import AbstractDACBenchAgent
from smac import Scenario
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue
from dacbench.abstract_agent import AbstractDACBenchAgent

class SMACAgent(AbstractDACBenchAgent):
    """Agent for Learning Rate in SGD Benchmark using SMAC"""

    def __init__(self, env, configspace, n_trials):
        """Initialize the Agent."""
        self.scenario = Scenario(configspace, deterministic=True, n_trials=n_trials)

        intensifier = HPOFacade.get_intensifier(
        self.scenario,
        max_config_calls=1,  # We basically use one seed per config only
    )

        def dummy_train(self, config: Configuration, seed: int = 0) -> float:
            pass

        self.smac = HPOFacade(
            self.scenario,
            dummy_train,  # We pass the target function here
            intensifier=intensifier,
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
            )
        
        

        super().__init__(env)

    def act(self, state=None, reward=None):
        """Returns the next action."""
        self.current_info = self.smac.ask()
        assert self.current_info.seed is not None

        return self.current_info.config["lr"]

    def train(self, state=None, reward=None):  # noqa: D102
        pass

    def end_episode(self, state=None, reward=None):  # noqa: D102
        value = TrialValue(cost=-reward, time=0.5)
        self.smac.tell(self.current_info, value=value)