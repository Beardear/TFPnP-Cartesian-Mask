# from collections import deque
import numpy as np
import random
import torch
import pickle as pickle
from torch.utils.data.dataloader import default_collate


class rpm:
    # replay memory
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        
    def append(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, env_batch, device):
        if self.size() < env_batch:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, env_batch)

        res = torch.stack(batch, dim=0)
        return res.to(device)
