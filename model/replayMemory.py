from collections import deque
import random

import torch
from torch.autograd import Variable  # The 'autograd' module is used to convert tensors into a variable.


# Class 'ReplayMemory':
#   Purpose:
#   Instance Variables:
#       'capacity': Maximum number of transitions that can be stored in replay memory
#       'memory': Actual storage for transitions with length 'capacity'
#   Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)  # The container datatype 'deque' enables fast appends and pops.

    # Method 'push':
    #   Purpose: Used to store a new transition in memory. In case, the number of stored transitions already equals
    #   'capacity', the oldest stored transition is automatically deleted from memory.
    #   Parameters:
    #       'transition': New transition to be stored
    def push(self, transition):
        self.memory.append(transition)

    # Method 'sample':
    #   Purpose: Providing a random choice of sample transitions for the agent to learn from.
    #   Parameters:
    #       'batch_size': Number of samples to return.
    #   Return:
    #       Map of tensors, with one element in the map per attribute of 'Transition', so 'state', 'new_state',
    #       'action', 'reward'. Each element in the map represents the value for an entire batch.
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        batch_map = map(lambda x: Variable(torch.cat(x, dim=0)), samples)
        return batch_map

    def has_batch_size(self, batch_size):
        return len(self.memory) >= batch_size
