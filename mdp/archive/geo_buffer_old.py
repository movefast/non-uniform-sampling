import itertools
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

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
    def __init__(self, capacity, alpha=None, beta=None, beta_increment=None):
        self.capacity = capacity
        self.memory = sliceable_deque([])
        self.position = 0
        self.geo_weights = np.array([1]*self.capacity).astype('float')
        self.loss_weights = np.array([0]*self.capacity).astype('float')
        
        self.num_ele = 0
        self.beta = beta or 0.4
        self.beta_increment_per_sampling = beta_increment or 0.001

    def add(self, *args):
        """Saves a transition."""
        if len(self.memory) == self.capacity:
            self.memory.popleft()
            # initialize discor weights
            self.loss_weights[:-1] = self.loss_weights[1:]
            self.loss_weights[-1] = 0
            self.geo_weights[:-1] = self.geo_weights[1:]
            self.geo_weights[-1] = 1
        else:
            self.loss_weights[len(self.memory)] = 0
            self.geo_weights[len(self.memory)] = 1
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
        # np.exp(-self.loss_weights * .9)
        if self.num_ele < self.capacity:
            weights = self.geo_weights[-self.num_ele:]
        else:
            weights = self.geo_weights
        # added min value for weights
        weights = np.clip(weights, 1e-3,None) 
        # TODO if add cer we need to think about how to adjust this
        weights_sum = self.geo_weights[-self.num_ele:].sum()
        # import pdb; pdb.set_trace()
        idxes = torch.multinomial(torch.from_numpy(weights).float(), n, replacement=False)
        # idxes = torch.multinomial(torch.from_numpy(weights).float(), n, replacement=True)
        batch = [self.memory[idx] for idx in idxes]
        
        sampling_probabilities = weights[idxes.numpy()] / weights_sum
        is_weight = np.power(len(self.memory) * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxes, is_weight

    def __len__(self):
        return len(self.memory)  
