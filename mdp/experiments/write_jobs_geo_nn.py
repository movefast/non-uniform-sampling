import pathlib
import random
from itertools import product

import numpy as np
import pandas as pd
from configs import ROOT_DIR
from fastprogress.fastprogress import master_bar, progress_bar

MAX_EVALS=200
count = 0


cur_dir = ROOT_DIR/"mdp"

def create_job(agent_type, hyper_params):
    global count
    cmd = f"python -m mdp.run_single_job --agent_type=\"{agent_type}\" --hyper_params=\"{hyper_params}\""
    with open(cur_dir/f"jobs/tasks_{count}.sh", 'w') as f:
        f.write(cmd)
    print(count, cmd)
    count += 1


def random_search(agent_type, param_grid, mb=master_bar(range(1)), max_evals=MAX_EVALS):#, num_runs=10):
    """Random search for hyperparameter optimization"""
    
    # Keep searching until reach max evaluations
    for j in mb:
        for i in progress_bar(range(max_evals),parent=mb):

            # Choose random hyperparameters
            hyper_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            print(hyper_params)
            # Evaluate randomly selected hyperparameters
            create_job(agent_type, hyper_params)


def grid_search(agent_type, param_grid):#, num_runs=10):
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
    # "PER_wo_Recency_Bias": None,
    "GNorm": None,
    "GEO": None,
    "GEO_V2": None,
    "GEO_V2_2": None,
    "GEO_V2_3": None,
    "GEO_V2_4": None,
    "CER": None,
#     "Diverse": DivAgent,
    # "Sarsa": None,
    "Sarsa_NN": None
    # "Sarsa_lambda": None,
    # "Meta_PER": None,
    # "Meta_CER": None,
}
agent_infos = {
    "Uniform": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10},
    "CER": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "k":1},
    # 'Diverse': {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "k":1},
    "PER": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":1e-4, "recency_bias": True, "grad_norm": False},
    "GNorm": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":1e-4, "recency_bias": True, "grad_norm": True},
    "GEO": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":1e-4, "p":.1},
    "GEO_V2": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_beta":0.4, "beta_increment":.00003, "num_meta_update":1, "weighting_strat":1, "lam":.5, "tau_1":1, "tau_2":1, "min_weight":1e-1},
    "GEO_V2_2": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_beta":0.4, "beta_increment":.00003, "num_meta_update":1, "weighting_strat":2, "lam":5, "tau_1":5, "min_weight":1e-1},
    "GEO_V2_3": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_beta":0.4, "beta_increment":.00003, "num_meta_update":1, "weighting_strat":3, "tau_1":1, "min_weight":1e-1},
    "GEO_V2_4": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_beta":0.4, "beta_increment":.00003, "num_meta_update":1, "weighting_strat":4, "tau_1":1, "min_weight":1e-1},
    # "Sarsa": {"step_size": .1, "buffer_size": 100, "batch_size": 1},
    "Sarsa_NN": {"step_size": .1},
    "Sarsa_lambda": {"step_size": .1, "buffer_size": 100, "batch_size": 1, "lambda":.9},
}
# param_grid = dict(
#     step_size=[1e-2,1e-3,3e-4],
# )

def get_lr(b=1e-2, a=2, n=5):
    return list(b/a**np.array(list(range(0, n))))


params_to_search = {
    "Uniform": {
        "step_size": get_lr(n=8),
        # "num_meta_update": [1, 2, 5, 10],
    },
    # "Sarsa_lambda": {
    #     "step_size": get_lr(1,n=8),
    #     "lambda":[0.99,0.98,0.96,0.9,0.84,0.68,0],
    # },
    "Sarsa_NN": {
        "step_size": get_lr(n=8),
        # "num_meta_update": [1, 2, 5, 10],
    },
    "CER": {
        "step_size": get_lr(n=8),
        # "num_meta_update": [1, 2, 5, 10],
    },
    "PER": {
        "step_size": get_lr(n=8),
        "buffer_alpha": get_lr(b=1.5,n=5), 
        "buffer_beta":get_lr(b=1,n=5),
#         "buffer_alpha": [0.6, 0.7], 
#         "buffer_beta":[0.4, 0.5, 0.6], 
        # "num_meta_update": [1, 2, 5, 10],
    },
#     "PER_wo_Recency_Bias": {
#         "step_size": get_lr(n=8),
#         "buffer_alpha": get_lr(b=1.5,n=5), 
#         "buffer_beta":get_lr(b=1,n=5),
# #         "buffer_alpha": [0.6, 0.7], 
# #         "buffer_beta":[0.4, 0.5, 0.6], 
#         # "num_meta_update": [1, 2, 5, 10],
#     },
    "GNorm": {
        "step_size": get_lr(n=8),
        "buffer_alpha": get_lr(b=1.5,n=5), 
        "buffer_beta":get_lr(b=1,n=5),
#         "buffer_alpha": [0.6, 0.7], 
#         "buffer_beta":[0.4, 0.5, 0.6], 
        # "num_meta_update": [1, 2, 5, 10],
    },
    "GEO": {
        "step_size": get_lr(n=8),
        "buffer_beta":get_lr(b=1,n=5), 
        "p": get_lr(.1,n=5),
        # "num_meta_update": [1, 2, 5, 10],
    },
    "GEO_V2": {
        "step_size": get_lr(n=8),
        "buffer_beta": get_lr(b=1,n=5), 
        "lam": get_lr(b=5,n=5), 
        "tau_1": get_lr(b=5,n=5),
        "tau_2": get_lr(b=5,n=5), 
        "min_weight": get_lr(b=.5,n=5),
        # "num_meta_update": [1, 2, 5, 10],
    },
    "GEO_V2_2": {
        "step_size": get_lr(n=8),
        "buffer_beta": get_lr(b=1,n=5), 
        "lam": get_lr(b=5,n=5), 
        "tau_1": get_lr(b=5,n=5),
        "min_weight": get_lr(b=.5,n=5),
        # "num_meta_update": [1, 2, 5, 10],
    },
    "GEO_V2_3": {
        "step_size": get_lr(n=8),
        "buffer_beta": get_lr(b=1,n=5), 
        "tau_1": get_lr(b=5,n=5),
        "min_weight": get_lr(b=.5,n=5),
        # "num_meta_update": [1, 2, 5, 10],
    },
    "GEO_V2_4": {
        "step_size": get_lr(n=8),
        "buffer_beta": get_lr(b=1,n=5), 
        "tau_1": get_lr(b=5,n=5),
        "min_weight": get_lr(b=.5,n=5),
        # "num_meta_update": [1, 2, 5, 10],
    },
    # "Meta_CER": {
    #     "meta_step_size": get_lr(b=1,a=10, n=5),
    #     "step_size": get_lr(n=8),
    #     "online_opt": ["sgd", "adam"],
    #     "num_meta_update": [1, 2, 5, 10],
    # },
    # "Meta_PER": {
    #     "meta_step_size": get_lr(b=1,a=10, n=5),
    #     "step_size": get_lr(n=8),
    #     "buffer_alpha": get_lr(b=1.5,n=5), 
    #     "buffer_beta":get_lr(b=1,n=5), 
    #     "online_opt": ["sgd", "adam"],
    #     "num_meta_update": [1, 2, 5, 10],
    # },
}


if __name__ == "__main__":
    for agent_type in master_bar(list(agents.keys())):
        print(agent_type)
        if agent_type == 'Meta_PER':
            random_search(agent_type, params_to_search[agent_type], max_evals=400)
        # elif agent_type in ('PER', 'GEO', "Meta_CER"):
        elif agent_type in ('GEO_V2', 'GEO_V2_2', 'GEO_V2_3', 'GEO_V2_4', "Meta_CER"):
            random_search(agent_type, params_to_search[agent_type])
        else:
            grid_search(agent_type, params_to_search[agent_type])

        print('Jobs: sbatch --array={}-{} ./mdp/jobs/run_cpu.sh'.format(0, count-1))
        
