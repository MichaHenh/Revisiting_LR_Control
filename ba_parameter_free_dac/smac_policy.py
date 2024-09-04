import sys
import numpy as np
import hydra

import exp_util as exp_util
from dacbench_custom.policy_agent import PolicyAgent
from dacbench_custom.custom_sgd_benchmark import CustomSGDBenchmark
from torch.optim import AdamW
from dacbench.abstract_benchmark import objdict
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="smacpolicy")
def evaluate_cost(cfg: DictConfig):
    r"""Function to meta-learn a learning rate policy using SMAC with hydra-smac-sweeper
    """

    env = CustomSGDBenchmark(AdamW, config=transform_to_objdict(cfg.dacbench_sgd_config)).get_environment()
    policy = PolicyAgent(env, [cfg['x{}'.format(i)] for i in range(len(cfg.hydra.sweeper.search_space.hyperparameters))])

    state, _ = env.reset()
    initial_loss = None
    current_loss = None
    total_cost = 0
    terminated, truncated = False, False
    reward = 0
    while not (terminated or truncated):
        action = policy.act(state, reward)
        print('step {}/{}'.format(env.c_step, env.n_steps))
        next_state, reward, terminated, truncated, _ = env.step(action)
        if initial_loss is None:
            initial_loss = env.loss
        current_loss = env.loss
        print('env step done')
        policy.train(next_state, reward)
        state = next_state

    total_cost = min((np.log(current_loss) - np.log(initial_loss)) / env.c_step, 0)
    print(current_loss)
    print('trial:')
    print('{}: {}'.format(policy, total_cost))

    return total_cost

def transform_to_objdict(config):
    cfg = objdict()

    for key in config:
        cfg[key] = config[key]

    return cfg

if __name__ == "__main__":
    sys.exit(evaluate_cost())  # pragma: no cover