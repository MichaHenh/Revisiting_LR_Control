import math
from dacbench.abstract_agent import AbstractDACBenchAgent
from smac import Scenario
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue
from dacbench.abstract_agent import AbstractDACBenchAgent

class PolicyAgent(AbstractDACBenchAgent):
    """Agent representing a Policy"""

    def __init__(self, env, policy):
        self.policy = policy
        self.act_counter = -1
        super().__init__(env)

    def act(self, state=None, reward=None):
        self.act_counter += 1
        return self.policy[self.act_counter]
    
    def reconfigure(self, policy):
        self.policy = policy

    def reset(self):
        self.act_counter = -1

    def train(self, state=None, reward=None):  # noqa: D102
        pass

    def end_episode(self, state=None, reward=None):  # noqa: D102
        pass

    def __str__(self) -> str:
        return str(self.policy)