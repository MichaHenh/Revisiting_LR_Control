import math
from dacbench.abstract_agent import AbstractDACBenchAgent
from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue
from dacbench.abstract_agent import AbstractDACBenchAgent
from ConfigSpace import Configuration

class SMACAgent(AbstractDACBenchAgent):
    r"""Agent for Learning Rate Training in DACBench using SMAC
        This is a hacky solution to train SMAC using the ask-and-tell interface with episode rewards.
        
    Args:
        - env: DACBench environment
        - configspace: ConfigurationSpace for hyperparameters to optimize
        - n_trials: number of trials to perform by SMAC
    """

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
        
        self.is_episode_beginning = True        

        super().__init__(env)

    def act(self, state=None, reward=None):
        """Returns the next action."""
        if self.is_episode_beginning:
            self.current_info = self.smac.ask()
            assert self.current_info.seed is not None
            self.is_episode_beginning = False

        return self.current_info.config["lr"]

    def train(self, state=None, reward=None):  # noqa: D102
        pass

    def end_episode(self, state=None, reward=None):  # noqa: D102
        value = TrialValue(cost=-reward, time=0.5)
        self.smac.tell(self.current_info, value=value)
        self.is_episode_beginning = True