from dacbench.runner import run_benchmark
from dacbench.agents import StaticAgent
from dacbench.logger import Logger
from pathlib import Path

from parameterfree import COCOB
from dacbench.abstract_benchmark import objdict
from dacbench_custom.custom_sgd_benchmark import CustomSGDBenchmark
from dacbench_custom.custom_tracking_wrapper import CustomTrackingWrapper

def setup_env(seed):
    cfg = objdict(
        {
        "optimizer_params": None,
        "cutoff": 1,
        "dataset_name": "CIFAR10",
        "torch_hub_model": ('pytorch/vision:v0.10.0', 'resnet18', False),
        "seed": seed
        }
    )
    # Get benchmark env
    bench = CustomSGDBenchmark(optimizer_type=COCOB, config=cfg)
    env = bench.get_environment()
    
    # Make logger to write results to file
    logger = Logger(experiment_name=f"cocob_cifar10_s{seed}", output_path=Path("results"))
    perf_logger = logger.add_module(CustomTrackingWrapper)
    
    env = CustomTrackingWrapper(env, logger=perf_logger)
    logger.set_env(env)
    logger.set_additional_info(seed=seed)
    
    return env, logger

for seed in range(1, 11):
    env, logger = setup_env(seed)
    
    # This could be any optimization or learning method
    agent = StaticAgent(env, [1])
    run_benchmark(env, agent, num_episodes=1, logger=logger)
