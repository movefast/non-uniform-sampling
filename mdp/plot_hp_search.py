import collections
import glob

import fire
import numpy as np
import pandas as pd
import torch
from configs import ROOT_DIR
from matplotlib import pyplot as plt

metrics = {"msbpe":{},"ve":{}, "all_reward_sums": {}, "hyper_params": {}}

# results = pd.DataFrame(columns = ['agent', 'score', 'params'],
#                                   index = list(range(MAX_EVALS)))

agent_names_in_plot={"NN": "Uniform", "NNP": "Uncertainty", "NNT":"Diversity"}
env_names_in_plot={'DoorWorldWide3':'GridWorldD3','DoorWorldWide11':'DoorWorldWide13X13D4'}
titles = {"msbpe":'MSPBE',"ve":'Value Error (VE)', "all_reward_sums":'Sum of Rewards'}
filtered_agent_list = ["Uniform", "PER", "GEO", "CER", "Sarsa_lambda"]
y_lims = {"msbpe","ve", "all_reward_sums"}
y_labels = {"msbpe":"MSPBE","ve":"Value Error (VE)", "all_reward_sums":"Sum of Rewards"}
stats_metric = {"msbpe":"AUC","ve":"AUC", "all_reward_sums":"Average Rewards"}

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
from mdp.run_single_job import num_runs
from mdp.write_jobs import params_to_search


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
    if metric_name == "all_reward_sums":
        print(algorithm, -np.around(np.mean(algorithm_means),decimals=4), np.around(np.std(np.mean(np.array(metrics[metric_name][env][algorithm]), axis=1))/np.sqrt(num_runs),decimals=4), sep='\t')
    else:
        print(algorithm, np.around(np.mean(algorithm_means)*300,decimals=4), np.around(np.std(np.mean(np.array(metrics[metric_name][env][algorithm]), axis=1)*300)/np.sqrt(num_runs),decimals=4), sep='\t')

    ax.plot(algorithm_means, label=env_names_in_plot.get(env,env)+'_'+agent_names_in_plot.get(algorithm, algorithm), alpha=0.5)
#     ax.set_ylim(0,.005)
    if metric_name == "msbpe":
        ax.set_ylim(0,.0015)
#     ax.set_ylim(-50,-5)
#     ax.set_ylim(0,.6)
    ax.fill_between(range(num_episodes), algorithm_means + algorithm_stds/np.sqrt(num_runs), algorithm_means - algorithm_stds/np.sqrt(num_runs), alpha=0.2)
    ax.legend()

for file in glob.glob(str(ROOT_DIR/'mdp/metrics/*')):
    dict_merge(metrics, torch.load(file))

def plot_param_search():
    for metric_name in ["msbpe","ve", "all_reward_sums"]:
        fig = plt.figure(figsize=(20,20))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        print(metric_name)
        print("agent", stats_metric[metric_name], "SEM", sep='\t')

        best_agent_names = []

        for env in env_infos:
            for i, agent_name in enumerate(filtered_agent_list):
                ax = fig.add_subplot(3, 2, i+1)
                agent_names = filter(lambda x: x.startswith(agent_name),  list(metrics[metric_name][env].keys()))
                metrics_slice = {agent_name: metrics[metric_name][env][agent_name] for agent_name in agent_names}
                sorted_agent_name_pairs = sorted([(np.mean(vals), algo) for algo, vals in metrics_slice.items()])
                
                for j, (_, algorithm) in enumerate(sorted_agent_name_pairs[:5]):
                    plot_metric(ax, env, algorithm, metric_name)
                    if j == 0:
                        best_agent_names.append(algorithm)
        fig.text(0.5, 0.04, 'Episodes', ha='center')
        fig.text(0.04, 0.5, titles[metric_name], va='center', rotation='vertical')

        fig.suptitle("Learning Rate Sweep")

        plt.savefig(ROOT_DIR/f'mdp/plots/params_{metric_name}.png')

        # plot best learning curves
        fig, ax = plt.subplots(figsize=(20,10))
        for algorithm in best_agent_names:
            plot_metric(ax, env, algorithm, metric_name)
        
        plt.ylabel(y_labels[metric_name],rotation=0, labelpad=20)
        plt.xlabel("Episodes")
        plt.title("Nonstationary Policy in 100-state RandomWalk (buffer_size=1000)")
        plt.savefig(ROOT_DIR/f'mdp/plots/{metric_name}.png')

def plot_best_learning_curves():
    for metric_name in ["msbpe","ve", "all_reward_sums"]:
        print(metric_name)
        print("agent", stats_metric[metric_name], "SEM", sep='\t')
        fig, ax = plt.subplots(figsize=(20,10))
        for env in env_infos:
            for i, agent_name in enumerate(filtered_agent_list):
                agent_names = filter(lambda x: x.startswith(agent_name),  list(metrics[metric_name][env].keys()))
                for algorithm in agent_names:
                    plot_metric(ax, env, algorithm, metric_name)

            plt.ylabel(y_labels[metric_name],rotation=0, labelpad=20)
            plt.xlabel("Episodes")
            # plt.ylim(0,.002)
            # plt.ylim(-150,-5)
            # plt.ylim(-800,-5)
            # plt.ylim(0,.006)
            # plt.plot([0,500],[-18,-18])
            # plt.title("Nonstationary Policy in 20-state RandomWalk (buffer_size=1000)")
            plt.title("Nonstationary Policy in 100-state RandomWalk (buffer_size=1000)")
            # plt.title("Random Policy in 20-state RandomWalk (buffer_size=1000)")

            # plt.legend()
            plt.savefig(ROOT_DIR/f'mdp/plots/{metric_name}.png')

def get_metric_stats(env, metric_name, algorithm):
    if metric_name == "all_reward_sums":
        algorithm_stats = -np.around(np.mean(metrics[metric_name][env][algorithm]),decimals=4)
        ste = np.around(np.std(np.mean(np.array(metrics[metric_name][env][algorithm]), axis=1))/np.sqrt(num_runs),decimals=4)
    else:
        algorithm_stats = np.around(np.mean(metrics[metric_name][env][algorithm])*300,decimals=4)
        ste = np.around(np.std(np.mean(np.array(metrics[metric_name][env][algorithm]), axis=1)*300)/np.sqrt(num_runs),decimals=4)
    return algorithm_stats, ste

import math


def plot_parameter_sensitivity():
    env = 'MDP'
    fig = plt.figure(figsize=(20,10))
    for metric_name in ["all_reward_sums"]:
        print(metric_name)
        unique_params = set([e for l in [list(v.keys()) for k, v in params_to_search.items() if k in filtered_agent_list] for e in l])
        for i, param in enumerate(unique_params):
            ax = fig.add_subplot(3, math.ceil(len(unique_params)/3), i+1)
            agent_list_for_param =  [agent_type for agent_type in filtered_agent_list if param in params_to_search[agent_type]]
            # try:
            #     agent_list_for_param.remove("Sarsa") 
            # except:
            #     pass
            for agent_type in agent_list_for_param:
                x_values = []
                y_values = []
                error_bars = []
                for val in params_to_search[agent_type][param]:
                    print(agent_type, param, val)
                    agent_names = list(filter(lambda x: x.startswith(agent_type) and f'{param}_{val}' in x,  list(metrics[metric_name][env].keys())))
                    print(agent_names)
                    metric_stats = [get_metric_stats(env, metric_name, agent_name) for agent_name in agent_names]
                    if metric_stats:
                        lst_of_stats, lst_of_stes = list(zip(*metric_stats))
                        x_values.append(val)
                        y_values.append(max(lst_of_stats))
                        error_bars.append(max(list(zip(lst_of_stats, lst_of_stes)))[1])
                print(x_values, y_values)
                ax.errorbar(x_values, y_values, label=agent_type, yerr=error_bars, capsize=5, elinewidth=1)#, markeredgewidth=10)


            ax.legend()
            ax.set_title(f'{param}')
            if param == "step_size":
                ax.set_xscale("log")
    plt.suptitle(f"Sensitivity Analysis in 100-state RandomWalk ({metric_name})")
    plt.savefig(ROOT_DIR/f'mdp/plots/{metric_name}_param_study.png')
            

def plot(plot_type):
    if plot_type == "params":
        plot_param_search()
    elif plot_type == "lc":
        plot_best_learning_curves()
    elif plot_type == "sensitivity":
        plot_parameter_sensitivity()


if __name__ == '__main__':
    fire.Fire(plot)


# def plot_parameter_sensitivity():
#     env = 'MDP'
#     for metric_name in ["all_reward_sums"]:
#         print(metric_name)
#         for param in set([e for l in [list(v.keys()) for k, v in params_to_search.items()] for e in l]):
#             fig = plt.figure(figsize=(20,10))
#             agent_list_for_param =  [agent_type for agent_type in filtered_agent_list if param in params_to_search[agent_type]]
#             for agent_type in agent_list_for_param:
#                 x_values = []
#                 y_values = []
#                 error_bars = []
#                 for val in params_to_search[agent_type][param]:
#                     print(agent_type, param, val)
#                     agent_names = list(filter(lambda x: x.startswith(agent_type) and f'{param}_{val}' in x,  list(metrics[metric_name][env].keys())))
#                     print(agent_names)
#                     metric_stats = [get_metric_stats(env, metric_name, agent_name) for agent_name in agent_names]
#                     if metric_stats:
#                         lst_of_stats, lst_of_stes = list(zip(*metric_stats))
#                         x_values.append(val)
#                         y_values.append(max(lst_of_stats))
#                         error_bars.append(max(list(zip(lst_of_stats, lst_of_stes)))[1])
#                 print(x_values, y_values)
#                 plt.plot(x_values,y_values, label=agent_type, alpha=1)
#                 plt.errorbar(x_values, y_values, yerr=error_bars, capsize=5, elinewidth=1)#, markeredgewidth=10)


#             plt.legend()
#             plt.title(f"Sensitivity Analysis in 100-state RandomWalk ({metric_name} | {param})")
#             plt.savefig(ROOT_DIR/f'mdp/plots/{metric_name}_{param}_param_study.png')
