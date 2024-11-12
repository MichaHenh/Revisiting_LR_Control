import sys
import numpy as np
import hydra

import exp_util as exp_util
from dacbench_custom.cosine_annealing_agent import CosineAnnealingWRAgent as CAWRAgent
from dacbench_custom.custom_sgd_benchmark import CustomSGDBenchmark
from torch.optim import AdamW
from dacbench.abstract_benchmark import objdict
from omegaconf import DictConfig
from time import time
from dacbench.runner import run_benchmark

@hydra.main(version_base=None, config_path="configs", config_name="smacpolicy")
def evaluate_cost(cfg: DictConfig):
    r"""Function to meta-learn a learning rate policy using SMAC with hydra-smac-sweeper
    """

    env = CustomSGDBenchmark(AdamW, config=transform_to_objdict(cfg.dacbench_sgd_config)).get_environment()
    agent = CAWRAgent(env, cfg.T0, cfg.eta_min, cfg.base_lr, cfg.t_mult)

    state, _ = env.reset()
    initial_loss = None
    current_loss = None
    total_cost = 0
    terminated, truncated = False, False
    reward = 0

    while not (terminated or truncated):
        ctime = time()
        action = agent.act(state, reward)
        # print("CAWR: {}".format(time()-ctime))
        # print('step {}/{}'.format(env.c_step, env.n_steps))
        next_state, reward, terminated, truncated, _ = env.step(action)
        # print("Env: {}".format(time()-ctime))
        if initial_loss is None:
            initial_loss = env.loss
        current_loss = env.loss
        # print('env step done')
        agent.train(next_state, reward)
        state = next_state
        # print("Total: {}".format(time()-ctime))

    total_cost = min((np.log(current_loss) - np.log(initial_loss)) / env.c_step, 0)
    print(current_loss)
    print('trial:')
    print('{}: {}'.format(agent, total_cost))

    return total_cost

def transform_to_objdict(config):
    cfg = objdict()

    for key in config:
        cfg[key] = config[key]

    return cfg

if __name__ == "__main__":
    sys.exit(evaluate_cost())  # pragma: no cover