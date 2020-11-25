import pathlib
import random
from itertools import product

import numpy as np
import pandas as pd
from configs import ROOT_DIR
from fastprogress.fastprogress import master_bar, progress_bar

MAX_EVALS=5
count = 0


cur_dir = ROOT_DIR/"mdp"

def create_job(agent_type, hyper_params):
    global count
    cmd = f"python -m mdp.run_single_job --agent_type=\"{agent_type}\" --hyper_params=\"{hyper_params}\""
    with open(cur_dir/f"jobs/tasks_{count}.sh", 'w') as f:
        f.write(cmd)
    print(count, cmd)
    count += 1


def random_search(agent_type, param_grid, mb=master_bar(range(1)), max_evals=MAX_EVALS, num_runs=10):
    """Random search for hyperparameter optimization"""
    
    # Keep searching until reach max evaluations
    for j in mb:
        for i in progress_bar(range(max_evals),parent=mb):

            # Choose random hyperparameters
            hyper_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            print(hyper_params)
            # Evaluate randomly selected hyperparameters
            create_job(agent_type, hyper_params)


def grid_search(agent_type, param_grid, num_runs=10):
    """grid search for hyperparameter optimization"""
    param_keys, values = zip(*param_grid.items())
    
    param_combos = [dict(zip(param_keys, combo)) for combo in product(*values)]
    
    mb = master_bar(param_combos)
    
    for i, hyper_params in enumerate(mb):
        print(hyper_params)
        create_job(agent_type, hyper_params)

agents = {
    "Uniform": None,
    "PER": None,
    "GEO": None,
    "CER": None,
#     "Diverse": DivAgent,
    "Sarsa": None,
}
agent_infos = {
    "Uniform": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10},
    "CER": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "k":1},
    'Diverse': {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "k":1},
    "PER": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":2e-4},
    "GEO": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":2e-4, "p":.1},
    "Sarsa": {"step_size": .1, "buffer_size": 100, "batch_size": 1},
}
param_grid = dict(
    step_size=[1e-2,1e-3,3e-4],
)

def get_lr(b=1e-2, n=2):
    return list(b/n**np.array(list(range(0, 5))))


params_to_search = {
    "Uniform": {
        "step_size": get_lr(),
    },
    "Sarsa": {
        "step_size": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
    },
    "CER": {
        "step_size": get_lr(),
    },
    "PER": {
        "step_size": get_lr(),
        "buffer_alpha": [0, 0.4, 0.6, 0.7, 0.8], 
        "buffer_beta":[0, 0.4, 0.5, 0.6, 1], 
#         "buffer_alpha": [0.6, 0.7], 
#         "buffer_beta":[0.4, 0.5, 0.6], 
    },
    "GEO": {
        "step_size": get_lr(),
        "buffer_alpha": [0, 0.4, 0.6, 0.7, 0.8], 
        "buffer_beta":[0, 0.4, 0.5, 0.6, 1], 
        "p": [.1, .3, .5, .7, .9],
    },
}


for agent_type in master_bar(list(agents.keys())):
    print(agent_type)
    if agent_type in ('PER', 'GEO'):
        random_search(agent_type, params_to_search[agent_type], max_evals=25)
    else:
        grid_search(agent_type, params_to_search[agent_type])

    print('Jobs: sbatch --array={}-{} ./run_cpu.sh'.format(0, count-1))
    
