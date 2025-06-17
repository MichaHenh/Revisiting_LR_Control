from dacbench.runner import run_benchmark
from dacbench.agents import StaticAgent
from dacbench.logger import Logger
from pathlib import Path
import math

from parameterfree.cocob_optimizer import COCOB
from parameterfree.cocob_trackable_optimizer import COCOBTrackable
from parameterfree.STORMplus import STORMplus
from parameterfree.DoWG import DoWG, CDoWG
from parameterfree.dadaptation import DAdaptAdam
from parameterfree.prodigy import Prodigy
from torch.optim import AdamW
from dacbench_custom.custom_sgd_benchmark import CustomSGDBenchmark
from dacbench_custom.custom_tracking_wrapper import CustomTrackingWrapper
from dacbench_custom.cosine_annealing_agent import CosineAnnealingWRAgent
from dacbench.abstract_benchmark import objdict
from dacbench_custom.smac_agent import SMACAgent
from dacbench_custom.policy_agent import PolicyAgent
from ConfigSpace import Configuration, ConfigurationSpace, Float

def get_optimizer_type(optimizer_type_name):
    match optimizer_type_name:
        case "ProdigyAdam":
            return Prodigy
        case "DAdaptAdam":
            return DAdaptAdam
        case "COCOB":
            return COCOB
        case "COCOB_trackable":
            return COCOBTrackable
        case "stormplus":
            return STORMplus
        case "dowg":
            return DoWG
        case "cdowg":
            return CDoWG
        case "adam":
            return AdamW
        
    return AdamW
        
def get_agent(agent, env, seed):
    match agent.type:
        case "StaticAgent":
            return StaticAgent(env, [agent.lr])
        case "CosineAnnealingWRAgent":
            return CosineAnnealingWRAgent(env, agent.T_0, agent.eta_min, agent.base_lr, agent.t_mult)
        case "PolicyAgent":
            return PolicyAgent(env, agent['policy_{}'.format(seed)])
        
    return StaticAgent(env [1])

def transform_to_objdict(config):
    cfg = objdict()

    for key in config:
        cfg[key] = config[key]

    return cfg

def setup_env(seed, cfg):

    if 'cutoff' in cfg:
        cfg.cutoff = int(cfg.cutoff)
    # Get benchmark env
    sgd_config = transform_to_objdict(cfg.dacbench_sgd_config)
    sgd_config['seed'] = seed
    bench = CustomSGDBenchmark(optimizer_type=get_optimizer_type(cfg.optimizer_type),
                               config=sgd_config)
    
    env = bench.get_environment()
    
    # Make logger to write results to file
    logger = Logger(experiment_name=".", output_path=Path('.'))
    perf_logger = logger.add_module(CustomTrackingWrapper)
    
    env = CustomTrackingWrapper(env, logger=perf_logger,
                                track_effective_lr= cfg.track_effective_lr if 'track_effective_lr' in cfg else False,
                                track_dlr=cfg.get('track_dlr'))
    logger.set_env(env)
    logger.set_additional_info(seed=seed)
    
    return env, logger


def run_smac(cfg, seed):
    sgd_config = transform_to_objdict(cfg.dacbench_sgd_config)
    sgd_config['seed'] = seed

    bench = CustomSGDBenchmark(optimizer_type=get_optimizer_type(cfg.optimizer_type),
                               config=sgd_config)
    
    env = bench.get_environment()

    cs = ConfigurationSpace(seed=seed)
    lr = Float("lr", (cfg.lr_min, cfg.lr_max), default=cfg.lr_default)
    cs.add_hyperparameters([lr])

    agent = SMACAgent(env, configspace=cs, n_trials=cfg.n_trials)

    run_benchmark(env, agent, num_episodes=cfg.n_trials)

    incumbent = agent.smac.optimize()
    print("Incumbent reached: {}".format(incumbent))
    return incumbent["lr"]


def run(cfg):
    if "smac" in cfg:
        incumbent = run_smac(cfg.smac, cfg.seed)
        env, logger = setup_env(cfg.seed, cfg)
        run_benchmark(env, StaticAgent(env, [incumbent]), num_episodes=cfg.num_episodes, logger=logger)
        return 100 if math.isnan(env.loss) else env.loss
    else:
        env, logger = setup_env(cfg.seed, cfg)
        run_benchmark(env, get_agent(cfg.agent, env, cfg.seed), num_episodes=cfg.num_episodes, logger=logger)
        return 100 if math.isnan(env.loss) else env.loss
