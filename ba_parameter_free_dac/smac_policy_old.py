import sys
import os
import time
import socket
import json
import hashlib
import numpy as np

import exp_util as exp_util
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter
from smac import AlgorithmConfigurationFacade as SMAC4AC
from smac.scenario import Scenario
from smac.initial_design import RandomInitialDesign
from dacbench_custom.policy_agent import PolicyAgent
from dacbench_custom.custom_sgd_benchmark import CustomSGDBenchmark
from dacbench.wrappers import PerformanceTrackingWrapper
from torch.optim import AdamW
from dacbench.abstract_benchmark import objdict
from dacbench.runner import run_benchmark

GLOBAL = None

class Globals:

    def __init__(self, env, policy, param_scale, result_cache):
        self.env = env
        self.policy = policy
        self.param_scale = param_scale
        self.result_cache = result_cache
        self.n_evals_to_n_steps = [0]
        self.initial_costs = {}

def evaluate_cost(cfg, seed):
    global GLOBAL

    # measure cost in steps
    # cutoff = GLOBAL.env.conditions[int(instance)][3]
    # GLOBAL.n_evals_to_n_steps.append(GLOBAL.n_evals_to_n_steps[-1] + cutoff)

    # Pre-processing on config
    # cfg = vectorize_config(cfg)
    # cfg = scale_vector(cfg, GLOBAL.param_scale)

    # Check result cache
    result_cache = GLOBAL.result_cache
    key = hashlib.md5('{}+{}'.format(cfg, seed).encode()).hexdigest()
    if result_cache is not None:
        value = result_cache.get(key)
        if value is not None:
            return value

    # Not in result cache, run target algorithm
    policy = GLOBAL.policy
    policy.reconfigure(cfg)
    policy.reset()
    env = GLOBAL.env
    
    # run_benchmark(env, policy, 1)
    # condition = env.conditions[int(instance)]
    # obs = env.conditioned_reset(True)
    # cutoff = condition[3]

    # GLOBAL.initial_costs[instance] = GLOBAL.initial_costs.get(instance, env.env.get_full_training_loss())
    # initial_loss = GLOBAL.initial_costs[instance]
    state, _ = env.reset()
    initial_loss = None
    current_loss = None
    total_cost = 0
    terminated, truncated = False, False
    reward = 0
    while not (terminated or truncated):
        action = policy.act(state, reward)
        next_state, reward, terminated, truncated, _ = env.step(action)
        if initial_loss is None:
            initial_loss = env.loss
        current_loss = env.loss
        policy.train(next_state, reward)
        state = next_state

        # policy.end_episode(state, reward)

    # total_cost = total_cost / 5 if (total_cost < 0 and not obs.get("crashed", 0)) else 0.0

    # if not obs.get("crashed", 0):
        # final_loss = env.env.get_full_training_loss()
    total_cost = min((np.log(current_loss) - np.log(initial_loss)) / env.c_step, 0)
    print(current_loss)
    print('trial: {}'.format(key))
    print('[{}] {}: {}'.format(key, policy, total_cost))

    if result_cache is not None:
        # store result in cache
        result_cache.store(key, total_cost)

    return total_cost

def transform_to_objdict(config):
    cfg = objdict()

    for key in config:
        cfg[key] = config[key]

    return cfg

def train_smac_policy(setup):
    global GLOBAL

   # start_time = time.time()

    # set up solution quality tracer / output directories for results
    os.makedirs(setup['result_dir'], exist_ok=True)
    #meta_info = {'start_time': start_time, 'host': socket.gethostname()}
    #sqt_id = exp_util.store_result(setup['result_dir'], setup, meta=meta_info, overwrite=setup['overwrite'])
    smac_output_dir = setup['result_dir']
    #'''os.path.join('''
    #''', '{}_smac'.format(sqt_id))'''



    # set up Globals (required by target runner)
    env = CustomSGDBenchmark(AdamW, config=transform_to_objdict(setup['dacbench_sgd_config'])).get_environment()
    policy = PolicyAgent(env, [])
    # param_scale = setup['param_scale']
    # cfg = policy.current_configuration()
    result_cache = None
    if setup.get("cache_evaluations", False):
        result_cache_path = setup['result_dir']
        #os.path.join(
        #, '{}.cache'.format(sqt_id))
        result_cache = exp_util.ResultCache(result_cache_path)
    GLOBAL = Globals(env, policy, None, result_cache)

    # set up SMAC

    # Build configuration space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    # weights = [UniformFloatHyperparameter("w{}".format(i), -10, 10) for i in range(5)]
    params = []
    x_min, x_max = setup['lr_min'], setup['lr_max']
    for i in range(setup['policy_length']):
        params.append(
            UniformFloatHyperparameter('{}'.format(i), x_min, x_max))
    cs.add_hyperparameters(params)

    # Build Scenario
    # scenario_dict = {}
    # scenario_dict["run_obj"] = "quality"  # we optimize quality (alternatively runtime)
    # scenario_dict["runcount-limit"] = setup['trials_train_limit']
    # scenario_dict["cs"] = cs
    # scenario_dict["deterministic"] = setup['deterministic']
    # scenario_dict["instances"] = [[i] for i in range(len(env.conditions))]
    # scenario_dict["output_dir"] = smac_output_dir
    # scenario_dict["limit_resources"] = False
    # scenario_dict["cost_for_crash"] = 1.0
    # scenario_dict["abort_on_first_run_crash"] = False
    scenario = Scenario(configspace=cs, output_directory=smac_output_dir, deterministic=setup['deterministic'],
                        n_trials=setup['trials_train_limit'])

    smac = SMAC4AC(scenario=scenario, target_function=evaluate_cost, overwrite=True)
    # run smac (training)
    print('--- TRAINING ---')
    incumbent = smac.optimize()
    print('--- END TRAINING ---')
    env.close()
    return incumbent

    # validate incumbents & store SQT
    '''
    print('--- VALIDATION ---')
    val_env = setup['val_env']
    GLOBAL.env = val_env
    GLOBAL.policy = val_env.policy_space()
    tracer = exp_util.SQTRecorder(time_label='episodes_trained', verbose=True)
    traj_json = os.path.join(smac_output_dir, 'run_{}'.format(setup["seed"]), 'traj.json')
    with open(traj_json) as file:
        for i, line in enumerate(file):
            entry = json.loads(line)
            n_evals = entry['evaluations']
            if n_evals <= 0:
                continue
            n_steps = GLOBAL.n_evals_to_n_steps[n_evals]
            # store policy
            inc_i = vectorize_config(entry['incumbent'])
            checkpoint_file = os.path.join(setup['result_dir'], sqt_id, '{}.npy'.format(i))
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            np.save(checkpoint_file, inc_i)
            # evaluate the policy
            costs = np.empty((len(val_env.conditions),))
            for j, cond_j in enumerate(val_env.conditions):
                costs[j] = evaluate_cost(entry['incumbent'], 0, str(j))
            f_val = np.mean(costs)
            tracer.log(n_evals, steps_trained=n_steps, f_val=f_val, inc_id=i)
    sqt = tracer.produce()
    exp_util.store_result(setup['result_dir'], setup, meta_info, sqt, overwrite=True)
    '''