import itertools
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Any

import numpy as np

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

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = sliceable_deque([])
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) == self.capacity:
            self.memory.popleft()
        self.memory.append(Transition(*args))

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

    def clear(self):
        self.memory = sliceable_deque([])
        self.position = 0

    def __len__(self):
        return len(self.memory)  
