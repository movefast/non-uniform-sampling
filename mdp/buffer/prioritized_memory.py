import random

import numpy as np
import torch
from mdp.buffer.sum_tree import SumTree
from mdp.replay_buffer import Transition


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01

    def __init__(self, capacity, alpha=None, beta=None, beta_increment=None, p=None):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        # self.geo_weights = np.geomspace(1, 256, num=capacity)
        self.p = p or 0.5
        self.geo_weights = np.flip([(self.p)*(1-self.p)**n for n in range(self.capacity)]).copy()
        self.weights_sum = self.geo_weights.sum()
        self.num_ele = 0
        self.a = alpha or 0.6
        self.beta = beta or 0.4
        self.beta_increment_per_sampling = beta_increment or 0.001

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, *args):
        p = self._get_priority(error)
        self.tree.add(p, Transition(*args))
        if self.num_ele < self.capacity:
            self.num_ele += 1

    def maxp(self):
        return self.tree.maxp()

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()

        # recency biased correction
        # data_idx = [idx - self.capacity + 1 for idx in idxs] 
        # if self.num_ele < self.capacity:
        #     weights = self.geo_weights[-self.num_ele:]
        #     weights_sum = self.geo_weights[-self.num_ele:].sum()
        # else:
        #     weights = self.geo_weights
        #     weights_sum = self.weights_sum
        # geo_sampling_probabilities = weights[np.array(data_idx)] / weights_sum
        # is_weight = np.power(geo_sampling_probabilities / sampling_probabilities, self.beta)

        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def last_n(self, seq_len):
        return self.tree.last_n(seq_len)

    # \beta^(t-s) / (\sum_{k=1}^t \beta^{t-k}) 
    def sample_geo(self, n):
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        if self.num_ele < self.capacity:
            weights = self.geo_weights[-self.num_ele:]
            weights_sum = self.geo_weights[-self.num_ele:].sum()
        else:
            weights = self.geo_weights
            weights_sum = self.weights_sum
        idxes = torch.multinomial(torch.from_numpy(weights).float(), n, replacement=False)
        batch = []
        idxs = []
        for x in idxes:
            (idx, p, data) = self.tree.get_by_idx(x)
            batch.append(data)
            idxs.append(idx)
        print(idxes,idxs)
        sampling_probabilities = weights[idxes.numpy()] / weights_sum
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def __len__(self):
        return self.tree.n_entries

    # per_agent_trace
    def last_n_idxes(self, seq_len):
        return self.tree.last_n_idxes(seq_len)

    def multiply_update(self, idx, multiplier):
        self.tree.multiply_update(idx, multiplier)
