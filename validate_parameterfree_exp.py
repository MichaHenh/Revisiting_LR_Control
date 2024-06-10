from dacbench.runner import run_benchmark
from dacbench.agents import StaticAgent
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.logger import Logger
from pathlib import Path

from torch.optim import AdamW
from parameterfree.parameter_free_sgd_benchmark import ParameterFreeSGDBenchmark

def setup_env(seed):
    # Get benchmark env
    bench = ParameterFreeSGDBenchmark(AdamW)
    env = bench.get_benchmark(seed=seed)
    
    # Make logger to write results to file
    logger = Logger(experiment_name=f"validate_parameterfree_s{seed}", output_path=Path("results"))
    perf_logger = logger.add_module(PerformanceTrackingWrapper)
    
    env = PerformanceTrackingWrapper(env, logger=perf_logger)
    logger.set_env(env)
    logger.set_additional_info(seed=seed)
    
    return env, logger

for seed in [1]:
    env, logger = setup_env(seed)
    
    # This could be any optimization or learning method
    agent = StaticAgent(env, [1e-3])
    run_benchmark(env, agent, num_episodes=30, logger=logger)
