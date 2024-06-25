from dacbench.runner import run_benchmark
from dacbench_custom.custom_tracking_wrapper import CustomTrackingWrapper
from dacbench.logger import Logger
from pathlib import Path
from dacbench_custom.custom_sgd_benchmark import CustomSGDBenchmark
from torch.optim import AdamW
from dacbench.abstract_benchmark import objdict
from dacbench.envs.policies.sgd_ca import CosineAnnealingAgent

def setup_env(seed):
    cfg = objdict(
        {
        "cutoff": 30,
        "seed": seed
        }
    )
    # Get benchmark env
    bench = CustomSGDBenchmark(AdamW, config=cfg)
    #bench = SGDBenchmark()
    env = bench.get_environment()
    
    # Make logger to write results to file
    logger = Logger(experiment_name=f"cosine_annealing_s{seed}", output_path=Path("results"))
    perf_logger = logger.add_module(CustomTrackingWrapper)
    
    env = CustomTrackingWrapper(env, logger=perf_logger)
    logger.set_env(env)
    logger.set_additional_info(seed=seed)
    
    return env, logger

for seed in range(1, 11):
    env, logger = setup_env(seed)
    
    # This could be any optimization or learning method
    agent = CosineAnnealingAgent(env, base_lr=1e-3, t_max=30000)
    run_benchmark(env, agent, num_episodes=1, logger=logger)
