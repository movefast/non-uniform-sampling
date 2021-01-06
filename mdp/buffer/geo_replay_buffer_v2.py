import itertools
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import torch


class Strat(IntEnum):
    TWO_EXP = 1
    ONE_EXP = 2
    DISCOR = 3
    GEO = 4


Trans = namedtuple('Transition', ('state', 'action', 'new_state', 'new_action', 'reward', 'discount'))

class Transition(Trans):
    __slots__ = ()  
    def __new__(cls, state, action, new_state, new_action, reward, discount=None):
        return super(Transition, cls).__new__(cls, state, action, new_state, new_action, reward, discount)

# @dataclass
# class Transition:
#     state: Any
#     action: Any
#     reward: Any
#     hidden: Any = None
#     discount: Any = 0.9


class sliceable_deque(deque):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start, index.stop, index.step))
        return deque.__getitem__(self, index)


class ReplayMemory(object):

    # alpha is the placeholder so that geoagent doesnt have to change a lot
    def __init__(self, capacity, alpha=None, beta=None, beta_increment=None, weighting_strat=Strat.GEO, lam=2, tau_1=5, tau_2=1, min_weight=1e-1):
        self.capacity = capacity
        self.memory = sliceable_deque([])
        self.position = 0
        self.geo_weights = np.array([0]*self.capacity).astype('float')
        self.loss_weights = np.array([0]*self.capacity).astype('float')
        
        self.num_ele = 0
        self.beta = beta or 0.4
        self.beta_increment_per_sampling = beta_increment or 0.001

        self.weighting_strat = weighting_strat
        self.lam = lam
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.min_weight = min_weight

    def add(self, *args):
        """Saves a transition."""
        if len(self.memory) == self.capacity:
            self.memory.popleft()
            # initialize discor weights
            self.loss_weights[:-1] = self.loss_weights[1:]
            self.loss_weights[-1] = 0
            self.geo_weights[:-1] = self.geo_weights[1:]
            self.geo_weights[-1] = 0
        else:
            self.loss_weights[len(self.memory)] = 0
            self.geo_weights[len(self.memory)] = 0 
        self.memory.append(Transition(*args))
        if self.num_ele < self.capacity:
            self.num_ele += 1

    def append_state(self, state):
        """Saves a transition."""
        if len(self.memory) == self.capacity:
            self.memory.popleft()
        self.memory.append(state)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    ##TODO (does not support sequence sampling in episodic task right now; needs to handle timestep v.s. episode)
    def sample_sequence(self, batch_size, seq_len):
        assert len(self.memory) > seq_len, "we don't have long enough trajectory to sample from"
        start_idxes = np.random.choice(len(self.memory)-seq_len, batch_size)
        end_idxes = start_idxes + seq_len
        return [self.memory[slice(start, end)] for (start, end) in zip(start_idxes, end_idxes)]

    def sample_successive(self, seq_len):
        assert len(self.memory) > seq_len, "we don't have long enough trajectory to sample from"
        start_idx = np.random.choice(len(self.memory)-seq_len)
        end_idx = start_idx + seq_len
        return self.memory[slice(start_idx, end_idx)]

    def last_n(self, seq_len):
        assert len(self.memory) > seq_len, "we don't have long enough trajectory to sample from"
        end_idx = len(self.memory)
        start_idx = end_idx - seq_len
        return self.memory[slice(start_idx, end_idx)]

        # \beta^(t-s) / (\sum_{k=1}^t \beta^{t-k}) 
    def sample_geo(self, n):
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        # old geo strat
        # if self.num_ele < self.capacity:
        #     weights = self.geo_weights[-self.num_ele:]
        # else:
        #     weights = self.geo_weights
        # # added min value for weights
        # weights = np.clip(weights, 1e-3,None) 
        # # TODO if add cer we need to think about how to adjust this
        # weights_sum = self.geo_weights[-self.num_ele:].sum()

        if self.weighting_strat == Strat.TWO_EXP:
            weights = (np.exp(-self.geo_weights/self.tau_1)+ self.lam * np.exp(-self.loss_weights/self.tau_2))/ (1 + self.lam)
        elif self.weighting_strat == Strat.ONE_EXP:
            weights = np.exp(-(self.geo_weights + self.lam * self.loss_weights)/self.tau_1)
        elif self.weighting_strat == Strat.DISCOR:
            weights = np.exp(-self.loss_weights/self.tau_1)
        elif self.weighting_strat == Strat.GEO:
            weights = np.exp(-self.geo_weights/self.tau_1) 
        else:
            raise NotImplementedError
        
        # other weighting strats I've tried
        # weights = np.exp(-(self.geo_weights + 5 * self.loss_weights)/5)
        # weights = np.exp(-(self.geo_weights/np.max(self.geo_weights)))
        # weights = (np.exp(-self.geo_weights/5)+ .2 * np.exp(-self.loss_weights))/1.2
        # weights = np.exp(-(self.geo_weights/(np.max(self.geo_weights) + 1e-3) + 1 * self.loss_weights/(np.max(self.loss_weights) + 1e-3)))

        weights = np.clip(weights, self.min_weight, None) 
        # default uniform
        # weights = np.ones(self.num_ele)
        if self.num_ele < self.capacity:
            weights = weights[-self.num_ele:]
        # TODO if add cer we need to think about how to adjust this
        weights_sum = weights[-self.num_ele:].sum()
        idxes = torch.multinomial(torch.from_numpy(weights).float(), n, replacement=False)
        # idxes = torch.multinomial(torch.from_numpy(weights).float(), n, replacement=True)
        batch = [self.memory[idx] for idx in idxes]
        
        sampling_probabilities = weights[idxes.numpy()] / weights_sum
        is_weight = np.power(len(self.memory) * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxes, is_weight

    def __len__(self):
        return len(self.memory)  
