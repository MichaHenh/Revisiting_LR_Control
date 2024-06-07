from dacbench.runner import run_benchmark
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.logger import Logger
from pathlib import Path
from dacbench.benchmarks import SGDBenchmark
from dacbench.envs.policies.sgd_ca import CosineAnnealingAgent

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

for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    env, logger = setup_env(seed)
    
    # This could be any optimization or learning method
    agent = CosineAnnealingAgent(env)
    run_benchmark(env, agent, num_episodes=30, logger=logger)
