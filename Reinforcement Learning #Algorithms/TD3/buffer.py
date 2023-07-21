import torch
import random
import numpy as np
from collections import deque # for replay buffer

class ReplayBuffer:

    def __init__(self, state_dim, action_dim , device, capacity):

        self.device   = device
        self.buffer = deque(maxlen=int(capacity))

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for state, action, reward, next_state, done in transitions:

            states.append(state.cpu().numpy())
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.cpu().numpy())
            dones.append(done)

        states      = torch.tensor(states)
        actions     = torch.tensor(actions)
        rewards     = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        dones       = torch.tensor(dones)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)
