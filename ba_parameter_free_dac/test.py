from dacbench.runner import run_benchmark
from dacbench.agents import StaticAgent
from dacbench.logger import Logger
from pathlib import Path

from parameterfree import COCOB
from parameterfree.cocob_trackable_optimizer import COCOBTrackable
from torch.optim import AdamW
from dacbench.benchmarks import SGDBenchmark
from dacbench_custom.custom_tracking_wrapper import CustomTrackingWrapper
from dacbench_custom.cosine_annealing_agent import CosineAnnealingWRAgent
from dacbench.abstract_benchmark import objdict


cfg = objdict()
cfg["epoch_mode"] =  False
cfg["cutoff"] = 20
cfg["dataset_name"] = 'CIFAR10'
cfg["torch_hub_model"] = ('pytorch/vision:v0.10.0', 'resnet18', False)


bench = SGDBenchmark(config=cfg)
    
env = bench.get_environment()
    
# Make logger to write results to file
logger = Logger(experiment_name="test", output_path=Path('./results'))
perf_logger = logger.add_module(CustomTrackingWrapper)

env = CustomTrackingWrapper(env, logger=perf_logger)
logger.set_env(env)

run_benchmark(env, StaticAgent(env, [0.001]), num_episodes=1, logger=logger)