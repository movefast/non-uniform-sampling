import random
import sys
from enum import IntEnum

import numpy as np
import torch
from mdp.replay_buffer import Transition


class Strat(IntEnum):
    PER = 1
    GEO = 2
    PER_GEO = 3 


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = sys.float_info.epsilon

    def __init__(self, capacity, alpha=None, beta=None, beta_increment=None, weighting_strat=None, lam=None, tau=None, min_weight=1e-1, geo_alpha=None, sim_mode=None):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.num_ele = 0

        self.geo_weights = np.array([0]*self.capacity).astype('float')
        self.min_weight = min_weight
        self.lam = lam
        self.a = alpha or 0.6
        self.beta = beta or 0.4
        self.beta_increment_per_sampling = beta_increment or 0.001
        self.weighting_strat = weighting_strat
        self.tau = tau
        self.geo_alpha = geo_alpha

        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        # 1)
        self.max_priority = self._get_priority(0)
        # 2)
        # self.max_priority = 1

        ## TODO: change this
        self.device = torch.device("cpu")
        self.sim_mode = sim_mode
        # self.sims = torch.zeros((self.capacity, self.capacity), dtype=float)
        self.sims = torch.eye(self.capacity, self.capacity, dtype=float)

    def add(self, error, *args):
        self.data[self.ptr] = Transition(*args) 
        # 1)
        # self.tree.set(self.ptr, self.max_priority) 
        # 2)
        self.tree.set(self.ptr, self._get_priority(error))
        self.geo_weights[self.ptr] = 0
        if self.sim_mode == 1: 
            self.sims[self.ptr] = 1
            self.sims[:, self.ptr] = 1
        elif self.sim_mode == 2: 
            self.sims[self.ptr] = 0
            self.sims[:, self.ptr] = 0
            self.sims[self.ptr, self.ptr] = 1
        self.ptr = (self.ptr + 1) % self.capacity
        self.num_ele = min(self.num_ele + 1, self.capacity)

    def maxp(self):
        return np.max(self.tree.nodes[-1])

    def set_sim(self, i, j, similarity):
        self.sims[i,j] = similarity

    def sample(self, n):
        batch = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        ind, weights = self.sample_from_tree(n)
        batch = self.data[ind]
        self.beta = min(self.beta + self.beta_increment_per_sampling, 1)
        return batch, ind, torch.FloatTensor(weights).to(self.device).reshape(-1)

    def test_sample(self, weighting_strat=None):
        if self.num_ele < self.capacity or self.ptr == self.num_ele -1:
            geo_last_idx = 0
        else:
            geo_last_idx = self.ptr + 1
        geo_weights = -self.geo_weights/(self.geo_weights[geo_last_idx]+ self.e)* (self.num_ele-1) *self.tau
        geo_weights = np.clip(np.exp(geo_weights), self.min_weight, None)
        if self.num_ele < self.capacity:
            geo_weights = geo_weights[:self.num_ele]
        # geo_weights /= geo_weights.sum()
        per_weights = self.tree.nodes[-1][:self.num_ele]
        per_weights_1 = per_weights / self.max_priority.item()
        if not weighting_strat:
            weighting_strat = self.weighting_strat
        if weighting_strat == Strat.PER:
            final_weights = per_weights_1
        elif weighting_strat == Strat.GEO:
            final_weights = geo_weights
        elif weighting_strat == Strat.PER_GEO:
            final_weights = (geo_weights + self.lam * per_weights_1)
        return final_weights
    
    def sample_from_tree(self, batch_size):
        if self.weighting_strat != Strat.PER:
            geo_weights = -self.geo_weights/(self.geo_weights.max()+self.e)*(self.num_ele-1)*self.tau
            geo_weights = np.clip(np.exp(geo_weights), self.min_weight, None)
            if self.num_ele < self.capacity:
                geo_weights = geo_weights[:self.num_ele]
        if self.weighting_strat != Strat.GEO:
            per_weights = self.tree.nodes[-1][:self.num_ele]
            per_weights_1 = per_weights / self.max_priority.item()
        if self.weighting_strat == Strat.PER:
            final_weights = per_weights_1
            batch_size = min(batch_size, np.count_nonzero(final_weights))
        elif self.weighting_strat == Strat.GEO:
            final_weights = geo_weights
        elif self.weighting_strat == Strat.PER_GEO:
            final_weights = (geo_weights + self.lam * per_weights_1)
        final_weights /= final_weights.sum()
        batch_size = min(batch_size, np.count_nonzero(final_weights))
        ind = np.random.choice(self.num_ele, batch_size, p=final_weights, replace=False)
        weights = np.array(final_weights[ind] ** -self.beta)
        weights /= weights.max()
        return ind, weights

    def batch_update(self, ind, priority):
        if self.weighting_strat != Strat.PER:
            geo_priority = self._get_geo_priority(priority).numpy()
            geo_priority /= geo_priority.max()
            geo_priority = 1 - geo_priority
            # 1)
            # temp = self.sims 
            # temp = torch.div(temp, temp.sum(dim=0,keepdim=True))
            # temp = torch.matrix_power(temp, 5)
            # temp = torch.clamp(torch.div(temp, temp.diagonal()[None,]), min=0, max=1)
            # geo_sims = (1 - torch.clamp(temp[ind], min=0, max=1).mean(axis=0)).numpy()
            # 2)
            # geo_sims = self.sims[ind].max(axis=0)
            # geo_sims = self.sims[ind].min(axis=0)
            # 3)
            # geo_sims = self.sims[self.ptr]
            # 4)
            if self.sim_mode == 1:
                geo_sims = (1 - self.sims[ind]).numpy()
            # 5)
            elif self.sim_mode == 2:
                geo_sims = self.sims[ind].numpy()
                # geo_sims = temp[ind].numpy()
            elif self.sim_mode == 0:
                geo_sims = np.ones(self.capacity)
            else:
                raise NotImplementedError
            if self.num_ele == self.capacity:
                self.geo_weights += geo_priority @ geo_sims
            else:
                self.geo_weights[:self.num_ele] += (geo_priority @ geo_sims)[:self.num_ele]
        # comment out to view per weights when testing
        # if self.weighting_strat != Strat.GEO:
        priority = self._get_priority(priority)
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)

    def update(self, ind, priority):
        priority = self._get_priority(priority)
        self.max_priority = max(priority, self.max_priority)
        self.tree.set(ind, priority)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def _get_geo_priority(self, error):
        return (np.abs(error) + self.e) ** self.geo_alpha

    def __len__(self):
        return self.num_ele


class SumTree(object):
    def __init__(self, max_size):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    def set(self, node_index, new_priority):
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]
        
        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2
