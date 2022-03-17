import numpy as np
from collections import namedtuple
import random


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer(object):
    def __init__(
        self,
        capacity,
        seed):

        self.capacity = capacity
        self.position = 0
        self.memory = []
        np.random.seed(seed)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)


def get_buffer(capacity, seed):
    return ReplayBuffer(capacity=capacity, seed=seed)
