from dacbench.runner import run_benchmark
from dacbench.agents import StaticAgent
from dacbench.logger import Logger
from pathlib import Path

from parameterfree import COCOB
from parameterfree.cocob_trackable_optimizer import COCOBTrackable
from torch.optim import AdamW
from dacbench_custom.custom_sgd_benchmark import CustomSGDBenchmark
from dacbench_custom.custom_tracking_wrapper import CustomTrackingWrapper
from dacbench_custom.cosine_annealing_agent import CosineAnnealingWRAgent
from dacbench.abstract_benchmark import objdict

def get_optimizer_type(optimizer_type_name):
    match optimizer_type_name:
        case "COCOB":
            return COCOB
        case "COCOB_trackable":
            return COCOBTrackable
        case "adam":
            return AdamW
        
    return AdamW
        
def get_agent(agent, env):
    match agent.type:
        case "StaticAgent":
            return StaticAgent(env, [agent.lr])
        case "CosineAnnealingWRAgent":
            return CosineAnnealingWRAgent(env, agent.T_0, agent.eta_min, agent.base_lr, agent.t_mult)
        
    return StaticAgent(env [1])

def transform_to_objdict(config):
    cfg = objdict()

    for key in config:
        cfg[key] = config[key]

    return cfg

def setup_env(seed, cfg):

    # Get benchmark env
    bench = CustomSGDBenchmark(optimizer_type=get_optimizer_type(cfg.optimizer_type),
                               config=transform_to_objdict(cfg.dacbench_sgd_config))
    
    env = bench.get_environment()
    
    # Make logger to write results to file
    logger = Logger(experiment_name=f"s{seed}", output_path=Path('.'))
    perf_logger = logger.add_module(CustomTrackingWrapper)
    
    env = CustomTrackingWrapper(env, logger=perf_logger,
                                track_effective_lr= cfg.track_effective_lr if 'track_effective_lr' in cfg else False)
    logger.set_env(env)
    logger.set_additional_info(seed=seed)
    
    return env, logger

def run(cfg):
    for seed in cfg.seeds:
        env, logger = setup_env(seed, cfg)

        run_benchmark(env, get_agent(cfg.agent, env), num_episodes=cfg.num_episodes, logger=logger)
