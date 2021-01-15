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
    # "Uniform": None,
    "PER_V2": None,
    "GEO_min_weight": None,
    "GEO_min_weight_adaptive_decay": None,
    "PER_GEO": None,
    "PER_WO_Recency_Bias_GEO": None,
}
# param_grid = dict(
#     step_size=[1e-2,1e-3,3e-4],
# )

def get_lr(b=1e-2, a=2, n=5):
    return list(b/a**np.array(list(range(0, n))))


params_to_search = {
    "PER_V2": {
        "step_size": get_lr(n=8),
        "per_alpha": get_lr(b=1,n=4), 
        "buffer_beta":get_lr(b=1,n=4),
    },
    "GEO_min_weight": {
        "step_size": get_lr(n=8),
        "buffer_beta":get_lr(b=1,n=4),
        "geo_alpha": [0], 
        "tau": get_lr(b=1,n=8),
        "min_weight": get_lr(b=0.2, n=5),
    },
    "GEO_min_weight_adaptive_decay": {
        "step_size": get_lr(n=8),
        "buffer_beta":get_lr(b=1,n=4),
        "geo_alpha": get_lr(b=1,n=4), 
        "tau": get_lr(b=1,n=8),
        "min_weight": get_lr(b=0.2, n=5),
    }, 
    "PER_GEO": {
        "step_size": get_lr(n=8),
        "per_alpha": get_lr(b=1,n=4), 
        "buffer_beta":get_lr(b=1,n=4),
        "geo_alpha": get_lr(b=1,n=4), 
        "tau": get_lr(b=1,n=8),
        "recency_bias": [True],
        "min_weight": get_lr(b=0.2, n=5),
        "lam": [0.5, 1, 2, 3, 5],
    },  
    "PER_WO_Recency_Bias_GEO": {
        "step_size": get_lr(n=8),
        "per_alpha": get_lr(b=1,n=4), 
        "buffer_beta":get_lr(b=1,n=4),
        "geo_alpha": get_lr(b=1,n=4), 
        "tau": get_lr(b=1,n=8),
        "recency_bias": [False],
        "min_weight": get_lr(b=0.2, n=5),
        "lam": [0.5, 1, 2, 3, 5],
    },
}

if __name__ == "__main__":
    for agent_type in master_bar(list(agents.keys())):
        print(agent_type)
        if agent_type == 'Meta_PER':
            random_search(agent_type, params_to_search[agent_type], max_evals=400)
        # elif agent_type in ('PER', 'GEO', "Meta_CER"):
        elif agent_type in ("GEO_min_weight", "GEO_min_weight_adaptive_decay", "PER_GEO", "PER_WO_Recency_Bias_GEO"):
            random_search(agent_type, params_to_search[agent_type])
        else:
            grid_search(agent_type, params_to_search[agent_type])

        print('Jobs: sbatch --array={}-{} ./mdp/jobs/run_cpu.sh'.format(0, count-1))
        