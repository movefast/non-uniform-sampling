import json
import os
import pathlib
import sys
import time
from copy import deepcopy

import agent
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from configs import ROOT_DIR
from fastprogress.fastprogress import master_bar, progress_bar
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm

from mdp.agents.cer_agent import LinearAgent as CERAgent
from mdp.agents.geo_agent import LinearAgent as GEOAgent
from mdp.agents.geo_agent_v2 import LinearAgent as GEOAgentV2
from mdp.agents.nn_agent import LinearAgent as NNAgent
from mdp.agents.per_agent import LinearAgent as PERAgent
from mdp.agents.sarsa_agent import LinearAgent as SarsaNNAgent
from mdp.archive.sarsa_agent import QLearningAgent as SarsaAgent
# from mdp.diversity_agent import LinearAgent as DivAgent
from mdp.env import policies
from mdp.env.errors import *
from mdp.env.policies import Policy
from mdp.env.random_walk import RandomWalk
from mdp.mdp_env import MazeEnvironment
from mdp.meta_cer_agent import LinearAgent as MCERAgent
from mdp.meta_per_agent import LinearAgent as MPERAgent
from mdp.sarsa_lambda import QLearningAgent as SarsaLAgent

gamma = .99
num_states = 100
num_runs = 25
num_episodes = 300
exp_decay_explor = True 


nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

cur_dir = ROOT_DIR/"mdp"
    
# random policy
def get_pred_error(agent, epsilon_greedy=False):
    # TODO: change 5 to num of state in the env
    rw_env = RandomWalk(num_states)
    X = np.eye(rw_env.states)
    X = np.vstack([X, np.zeros((1,rw_env.states))])
    if type(agent) is SarsaAgent or type(agent) is SarsaLAgent:
        action_values = torch.from_numpy(agent.q[1:]).float()
    else:
        # 1 linear agent
        # action_values = agent.nn.i2o.weight.t()[1:]
        # 2) nn agent
        state_features = np.eye(num_states+2).astype("float32")
        action_values = agent.nn(torch.from_numpy(state_features))[1:]
    state_action_pol = F.softmax(action_values, dim=1)
    if not epsilon_greedy:
        policy = policies.fromActionArray([.5,.5])
    else:
        policy = torch.zeros_like(action_values)
        # 1) on-policy
        policy.scatter_(1, action_values.max(1)[1].unsqueeze(-1), (1 - agent.epsilon))
        policy += agent.epsilon / 2
        policy = policies.fromStateArray(policy.cpu().detach().numpy())
        
    
    P = rw_env.buildTransitionMatrix(policy)
    R =rw_env.buildAverageReward(policy)
    d = rw_env.getSteadyStateDist(policy)
    
    AbC = partiallyApplyMSPBE(X, P, R, d, gamma)
    theta = action_values @ torch.tensor([.5, .5])
    if epsilon_greedy:
        theta = (1 - agent.epsilon) * action_values.max(1)[0] + agent.epsilon * theta
#         theta = (action_values * state_action_pol).sum(dim=1)
    theta = theta.detach().cpu().numpy()
    mspbe = MSPBE(theta[:-1], *AbC)
    return mspbe, d


def run_episode(env, agent, state_visits=None, keep_history=False):
    is_terminal = False
    step_count = 0
    
    obs = env.env_start(keep_history=keep_history)
    action = agent.agent_start(obs)
    
    if state_visits is not None:
        state_visits[obs[0]] += 1

    while not is_terminal:
        reward, obs, is_terminal = env.env_step(action)
#         print(agent.steps,end='\r')
        step_count += 1
        state = obs
        if step_count == 1000:
            agent.agent_end(reward, state, append_buffer=False)
            break
        elif is_terminal:
            agent.agent_end(reward, state, append_buffer=True)
        else:
            action = agent.agent_step(reward, state)

        if state_visits is not None:
            state_visits[state[0]] += 1
    
    targets = np.flip([gamma**n for n in range(1,num_states+1)]).copy()# * (1 - agent.epsilon / 2)
    if type(agent) is SarsaAgent or type(agent) is SarsaLAgent:
        predictions = agent.q.max(axis=1)[1:-1] * (1 - agent.epsilon) \
            + agent.epsilon * agent.q.min(axis=1)[1:-1]
    else:
        # 1) linear agent
        # predictions = agent.nn.i2o.weight.t().max(1)[0][1:-1].detach().numpy() * (1 - agent.epsilon) \
        #     + agent.epsilon * agent.nn.i2o.weight.t().mean(dim=1)[1:-1].detach().numpy()
        # 2) nn agent
        state_features = np.eye(num_states+2).astype("float32")
        action_values = agent.nn(torch.from_numpy(state_features))
        predictions = action_values.max(1)[0][1:-1].detach().numpy() * (1 - agent.epsilon) \
            + agent.epsilon * action_values.mean(dim=1)[1:-1].detach().numpy()

    msbpe, d = get_pred_error(agent, epsilon_greedy=True)
    ve = np.sum(np.array(d[:-1]) * (predictions-targets)**2)
    if keep_history:
        history = env.history
        env.env_cleanup()
        return step_count, msbpe, ve, history
    else:
        return step_count, msbpe, ve
    

envs = {
    'Grid-World': MazeEnvironment,
}


env_infos = {
    'MDP': {
        "maze_dim": [1, num_states+2], 
        "start_state": [0, num_states//2+1], 
        "end_state": [0, num_states+1],
        "obstacles":[],
        "doors": {tuple():[]},
    },
}


agents = {
    "Uniform": NNAgent,
    "PER": PERAgent,
    "PER_wo_Recency_Bias": PERAgent,
    "GNorm": PERAgent,
    "GEO": GEOAgent,
    "GEO_V2": GEOAgent,
    "GEO_V2_2": GEOAgentV2,
    "GEO_V2_3": GEOAgentV2,
    "GEO_V2_4": GEOAgentV2,
    "CER": CERAgent,
    "Sarsa": SarsaAgent,
    "Sarsa_NN": SarsaNNAgent,
    "Sarsa_lambda": SarsaLAgent,
    "Meta_PER": MPERAgent,
    "Meta_CER": MCERAgent,
}
agent_infos = {
    # "Sarsa": {"step_size": .1, "buffer_size": 100, "batch_size": 1},
    "Sarsa_NN": {"step_size": .1, "num_meta_update":1},
    "Sarsa_lambda": {"step_size": .1, "buffer_size": 100, "batch_size": 1, "lambda":.9},
    "Uniform": {"step_size": 1e-2, "buffer_size": 1000, "batch_size": 10, "num_meta_update":1},
    "CER": {"step_size": 1e-2, "buffer_size": 1000, "batch_size": 10, "k":1, "num_meta_update":1},
    "PER": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":1e-4, "recency_bias": True, "grad_norm": False, "num_meta_update":1},
    "GNorm": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":1e-4, "recency_bias": True, "grad_norm": True, "num_meta_update":1},
    "PER_wo_Recency_Bias": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":1e-4, "recency_bias": False, "grad_norm": False, "num_meta_update":1},
    "GEO": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":1e-4, "p":.1, "num_meta_update":1},
    "GEO_V2": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_beta":0.4, "beta_increment":1e-4, "num_meta_update":1, "weighting_strat":1, "lam":.5, "tau_1":1, "tau_2":1, "min_weight":1e-1},
    "GEO_V2_2": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_beta":0.4, "beta_increment":1e-4, "num_meta_update":1, "weighting_strat":2, "lam":5, "tau_1":5, "min_weight":1e-1},
    "GEO_V2_3": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_beta":0.4, "beta_increment":1e-4, "num_meta_update":1, "weighting_strat":3, "tau_1":1, "min_weight":1e-1},
    "GEO_V2_4": {"step_size": 3e-3, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_beta":0.4, "beta_increment":1e-4, "num_meta_update":1, "weighting_strat":4, "tau_1":1, "min_weight":1e-1},
    "Meta_PER": {"step_size": 1e-2, "meta_step_size": 1e-1, "buffer_size": 1000, "batch_size": 10, "correction":True, "buffer_alpha":0.6, "buffer_beta":0.4, "beta_increment":1e-4, "num_meta_update":1},
    "Meta_CER": {"step_size": 1e-2, "meta_step_size": 1e-1, "buffer_size": 1000, "batch_size": 10, "k":1, "num_meta_update":1},
}


all_state_visits = {} # Contains state visit counts during the last 10 episodes
all_history = {}
metrics = {"msbpe":{},"ve":{}, "all_reward_sums": {}, "hyper_params": {}, "states": {}}


def objective(agent_type, hyper_params, num_runs=num_runs):
    start = time.time()
    
    Environment = envs['Grid-World']
    
    all_state_visits = {} # Contains state visit counts during the last 10 episodes
    all_history = {}

    mb = master_bar(env_infos.items())
    algorithm = agent_type + '_' + '_'.join([f'{k}_{v}' for k, v in hyper_params.items()])
    
    for env_name, env_info in mb:
        print(env_name)
        if env_name not in all_state_visits:
            all_state_visits[env_name] = {}
        for metric in metrics:
            if env_name not in metrics[metric]:
                metrics[metric][env_name] = {}
                metrics[metric][env_name].setdefault(algorithm, [])
            
        all_state_visits[env_name][algorithm] = []
        for run in progress_bar(range(num_runs), parent=mb):
            agent = agents[agent_type]()
            env = Environment()
            env.env_init(env_info)
#             print(env_info)
            agent_info = {"num_actions": 2, "num_states": env.cols * env.rows, "epsilon": 1, "step_size": 0.1, "discount": gamma} 
            agent_info["seed"] = run
            agent_info.update(agent_infos[agent_type])
            agent_info.update(hyper_params)
#             if algorithm in ('PER', 'GEO'):
#                 beta_increment = (1 - agent_info['buffer_beta']) / (num_episodes - 100)
            np.random.seed(run)
            agent.agent_init(agent_info)

            reward_sums = []
            lst_of_msbpe = []
            lst_of_ve = []
            lst_of_sampled_states = []
            state_visits = np.zeros(env.cols * env.rows)
            if exp_decay_explor:
                epsilon = 1
            else:
                epsilon = .1
            agent.per_power = 1
            for episode in range(num_episodes):
                print(f"episode {episode}",end='\r')
                agent.epsilon = epsilon
                if episode < num_episodes - 10:
                    episode_start = time.time()
                    sum_of_rewards, msbpe, ve = run_episode(env, agent) 
                    episode_end = time.time()
                    if episode_end - episode_start > 15:
                        return False
                else: 
#                   Runs an episode while keeping track of visited states and history
                    sum_of_rewards, msbpe, ve, history = run_episode(env, agent, state_visits, keep_history=True)
                    all_history.setdefault(env_name, {}).setdefault(algorithm, []).append(history)
                if exp_decay_explor:
                    epsilon *= 0.99
                reward_sums.append(sum_of_rewards)
                lst_of_msbpe.append(msbpe)
                lst_of_ve.append(ve)
                lst_of_sampled_states.append(agent.sampled_state.tolist())
                agent.sampled_state = np.zeros(env.cols * env.rows)

            metrics["all_reward_sums"][env_name].setdefault(algorithm, []).append(reward_sums)
            all_state_visits[env_name].setdefault(algorithm, []).append(state_visits)
            metrics["msbpe"][env_name].setdefault(algorithm, []).append(lst_of_msbpe)
            metrics["ve"][env_name].setdefault(algorithm, []).append(lst_of_ve)
            metrics["states"][env_name].setdefault(algorithm, []).append(lst_of_sampled_states)

    end = time.time()
    print(end - start)
    metrics['hyper_params'][algorithm] = hyper_params
    torch.save(metrics, cur_dir/f'metrics/{algorithm}.torch')
    return algorithm, np.mean(metrics["all_reward_sums"][env_name][algorithm]), hyper_params


if __name__ == '__main__':
    fire.Fire(objective)
