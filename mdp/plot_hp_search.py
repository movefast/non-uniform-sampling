import collections
import glob

import numpy as np
import pandas as pd
import torch
from configs import ROOT_DIR
from matplotlib import pyplot as plt

metrics = {"msbpe":{},"ve":{}, "all_reward_sums": {}, "hyper_params": {}}

# results = pd.DataFrame(columns = ['agent', 'score', 'params'],
#                                   index = list(range(MAX_EVALS)))

tmp={"NN": "Uniform", "NNP": "Uncertainty", "NNT":"Diversity"}
tmp1={'DoorWorldWide3':'GridWorldD3','DoorWorldWide11':'DoorWorldWide13X13D4'}
titles = {"msbpe":'MSPBE',"ve":'Value Error (VE)', "all_reward_sums":'Sum of Rewards'}

env_infos = {
    'MDP': {
        "maze_dim": [1, 102], 
        "start_state": [0, 51], 
        "end_state": [0, 101],
        "obstacles":[],
        "doors": {tuple():[]},
    },
}

num_episodes = 300

from run_single_job import num_runs


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        elif k in dct and isinstance(dct[k], list) and isinstance(v, list):
            print('hehehe')
        else:
            dct[k] = merge_dct[k]


def plot_metric(ax, env, algorithm, metric_name):
    if metric_name == "all_reward_sums":
        algorithm_means = -np.mean(metrics[metric_name][env][algorithm], axis=0)
    else:
        algorithm_means = np.mean(metrics[metric_name][env][algorithm], axis=0)
    algorithm_stds = np.std(metrics[metric_name][env][algorithm], axis=0)
    ax.plot(algorithm_means, label=tmp1.get(env,env)+'_'+tmp.get(algorithm, algorithm),
             alpha=0.5)
#     ax.set_ylim(0,.005)
    if metric_name == "msbpe":
        ax.set_ylim(0,.0015)
#     ax.set_ylim(-50,-5)
#     ax.set_ylim(0,.6)
    ax.fill_between(range(num_episodes), algorithm_means + algorithm_stds/np.sqrt(num_runs), algorithm_means - algorithm_stds/np.sqrt(num_runs), alpha=0.2)
    ax.legend()

for file in glob.glob(str(ROOT_DIR/'mdp/metrics/*')):
    dict_merge(metrics, torch.load(file))

filtered_agent_list = ["Uniform", "PER", "GEO", "CER", "Sarsa"]
y_lims = {"msbpe","ve", "all_reward_sums"}


for metric_name in ["msbpe","ve", "all_reward_sums"]:
    fig = plt.figure(figsize=(20,20))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for env in env_infos:
        for i, agent_name in enumerate(filtered_agent_list):
            ax = fig.add_subplot(3, 2, i+1)
            agent_names = filter(lambda x: x.startswith(agent_name),  list(metrics[metric_name][env].keys()))
            metrics_slice = {agent_name: metrics[metric_name][env][agent_name] for agent_name in agent_names}
            sorted_agent_name_pairs = sorted([(np.mean(vals), algo) for algo, vals in metrics_slice.items()])
            for _, algorithm in sorted_agent_name_pairs[:5]:
                plot_metric(ax, env, algorithm, metric_name)
    fig.text(0.5, 0.04, 'Episodes', ha='center')
    fig.text(0.04, 0.5, titles[metric_name], va='center', rotation='vertical')

    fig.suptitle("Learning Rate Sweep")



    plt.savefig(ROOT_DIR/f'mdp/plots/{metric_name}.png')
