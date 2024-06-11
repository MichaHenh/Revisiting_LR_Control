from dacbench.runner import run_benchmark
from dacbench.agents import StaticAgent
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.logger import Logger
from pathlib import Path

from parameterfree import COCOB
from dacbench_custom.custom_sgd_benchmark import CustomSGDBenchmark
from dacbench_custom.custom_tracking_wrapper import CustomTrackingWrapper

def setup_env(seed):
    # Get benchmark env
    bench = CustomSGDBenchmark(optimizer_type=COCOB)
    env = bench.get_benchmark(seed=seed)
    
    # Make logger to write results to file
    logger = Logger(experiment_name=f"cocob_s{seed}", output_path=Path("results"))
    perf_logger = logger.add_module(CustomTrackingWrapper)
    
    env = CustomTrackingWrapper(env, logger=perf_logger)
    logger.set_env(env)
    logger.set_additional_info(seed=seed)
    
    return env, logger

for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    env, logger = setup_env(seed)
    
    # This could be any optimization or learning method
    agent = StaticAgent(env, [1])
    run_benchmark(env, agent, num_episodes=30, logger=logger)
