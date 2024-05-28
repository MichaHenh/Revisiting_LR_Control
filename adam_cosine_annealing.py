from dacbench.runner import run_benchmark
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.logger import Logger
from pathlib import Path
from dacbench.abstract_agent import AbstractDACBenchAgent
from dacbench.benchmarks import SGDBenchmark
from torch.optim.lr_scheduler import CosineAnnealingLR

def setup_env(seed):
    # Get benchmark env
    bench = SGDBenchmark()
    env = bench.get_benchmark(seed=seed)
    
    # Make logger to write results to file
    logger = Logger(experiment_name=f"cosine_annealing_s{seed}", output_path=Path("results"))
    perf_logger = logger.add_module(PerformanceTrackingWrapper)
    
    env = PerformanceTrackingWrapper(env, logger=perf_logger)
    logger.set_env(env)
    logger.set_additional_info(seed=seed)
    
    return env, logger


class CosineAnnealingAgent(AbstractDACBenchAgent):
    def __init__(self, env):
        """Initialize the generic agent."""
        self.schedule = CosineAnnealingLR()
        self.env = env

    def act(self, state, reward):
        """Returns action."""
        return self.policy(self.env, state)

    def train(self, next_state, reward):
        """Train agent."""

    def end_episode(self, state, reward):
        """End episode."""



for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    env, logger = setup_env(seed)
    
    # This could be any optimization or learning method
    agent = CosineAnnealingAgent(env)
    run_benchmark(env, agent, num_episodes=10, logger=logger)
