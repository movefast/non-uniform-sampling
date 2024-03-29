import numpy as np


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.error = np.zeros(2 * capacity - 1)
        self.visits = np.zeros(2 * capacity - 1)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def maxp(self):
        return np.max(self.tree[self.capacity -1:])

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.visits[idx] = 0
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # # update priority
    # def update(self, idx, p):
    #     # TODO comment out visit discount logic
    #     self.visits[idx] += 1
    #     # p /= np.sqrt(self.visits[idx])

    #     last_exp_idx = self.write + self.capacity - 1
    #     self.error[idx] = p
    #     if idx == 2 * self.capacity - 2:
    #         next_idx = self.capacity - 1
    #     else:
    #         next_idx = idx + 1
    #     if next_idx == last_exp_idx:
    #         p = self.error[idx] 
    #     else:
    #         p = self.error[idx] / (1 + self.error[next_idx] * 10)
        


    #     change = p - self.tree[idx]

    #     # p = 1

    #     self.tree[idx] = p
    #     self._propagate(idx, change)

    # update prev value
    def update(self, idx, p):
        # TODO comment out visit discount logic
        self.visits[idx] += 1
        # p /= np.sqrt(self.visits[idx])

        last_exp_idx = self.write + self.capacity - 1
        self.error[idx] = p
        if idx == self.capacity - 1 :
            prev_idx = 2 * self.capacity - 2
        else:
            prev_idx = idx - 1
        if prev_idx == last_exp_idx - 1:
            p = self.error[prev_idx] 
        else:
            p = self.error[prev_idx] / (1 + self.error[idx] * 100)

        change = p - self.tree[prev_idx]

        # p = 1

        self.tree[prev_idx] = p
        self._propagate(prev_idx, change)

        if idx == 2 * self.capacity - 2:
            next_idx = self.capacity - 1
        else:
            next_idx = idx + 1
        if next_idx == last_exp_idx:
            p = self.error[idx] 
        else:
            p = self.error[idx] / (1 + self.error[next_idx] * 100)
        


        change = p - self.tree[idx]

        # p = 1

        self.tree[idx] = p
        self._propagate(idx, change)

        

    def update_value(self, idx, val):
        dataIdx = idx - self.capacity + 1
        self.data[dataIdx] = val

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def get_by_idx(self, dataIdx):
        idx = dataIdx + self.capacity - 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def last_n(self, seq_len):
        assert self.n_entries > seq_len, "we don't have long enough trajectory to sample from"
        dataIdx = self.write -1
        results = []
        idxs = []
        last_idxs = []
        for _ in range(seq_len):
            if dataIdx < 0:
                dataIdx = self.capacity - 1
            idx = dataIdx + self.capacity - 1
            # results.append((idx, self.tree[idx], self.data[dataIdx]))
            # consider as on policy
            idxs.append(idx)
            results.append(self.data[dataIdx])
            last_idxs.append(-1 if dataIdx == 0 else dataIdx - 2 + self.capacity)
            dataIdx -= 1
        return results, idxs, np.ones([seq_len]),  last_idxs

    def last_n_idxes(self, seq_len):
        assert self.n_entries >= seq_len, "we don't have long enough trajectory to sample from"
        dataIdx = self.write -1
        idxs = []
        for _ in range(seq_len):
            if dataIdx < 0:
                dataIdx = self.capacity - 1
            idx = dataIdx + self.capacity - 1
            idxs.append(idx)
            dataIdx -= 1
        return idxs

    def multiply_update(self, idx, multiplier):
        # TODO comment out visit discount logic
        self.visits[idx] += 1
        # p /= np.sqrt(self.visits[idx])
        p = self.tree[idx] * multiplier
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)
