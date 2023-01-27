import gym
import sys
import math
import random
import collections
import numpy as np
from time import sleep
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Actor
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 24)  # Input : state
        self.fc2 = nn.Linear(24, 2)  # Output : Softmax policy for action distribution

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        policy_distribution = F.softmax(x, dim = -1)

        return policy_distribution # x policy with softmax prob

# Critic
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)  # Input : state
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)  # Output : Appoximated Value

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        Approxed_value = self.fc3(x)

        return Approxed_value

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen = 50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, number):
        mini_batch = random.sample(self.buffer, number)
        states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, terminated, truncated = transition

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)
            truncateds.append(truncated)

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

def soft_update(Target_Network, Current_Network):
    for target_param, current_param in zip(Target_Network.parameters(), Current_Network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - beta) + current_param.data * beta)

memory= ReplayBuffer()
alpha = 0.01
beta  = 0.5 # Update Weight
gamma = 0.99
episode = 0
MAX_EPISODE = 10000

pi = PolicyNetwork().to(device)
V  = ValueNetwork().to(device)
V_target = ValueNetwork().to(device)

V_target.load_state_dict(V.state_dict())

pi_optimizer = optim.Adam(pi.parameters(), lr = alpha)
V_optimizer = optim.Adam(V.parameters(), lr = alpha)

def train(memory, V, V_target, V_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = memory.sample(64)

    critic_loss = 0
    actor_loss  = 0

    # Critic Update
    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
        if terminated or truncated:
            y = reward
        else:
            y = reward + gamma * V_target(next_state)

        critic_loss += (y - V(state))**2

    critic_loss = critic_loss / 64
    V_optimizer.zero_grad()
    critic_loss.backward()

    V_optimizer.step()

    # Actor Update
    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
        actor_loss += (reward + (gamma * V(next_state)) - V(state)) * ((pi(state)[action] + 1e-5).log())

    actor_loss = -actor_loss / 64 # Gradient Ascent

    pi_optimizer.zero_grad()
    actor_loss.backward()
    pi_optimizer.step()

#env = gym.make('CartPole-v1', render_mode = 'human')
env = gym.make('CartPole-v1') #, render_mode = 'human')

while episode < MAX_EPISODE:

    state = env.reset()
    state = torch.tensor(state[0]).float().to(device)
    score = 0

    terminated = False
    truncated  = False

    # Make Experiences
    while True:
        policy = pi(state)
        action = torch.multinomial(policy, 1).item()
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = torch.tensor(next_state).float().to(device)
        memory.put((state, action, reward, next_state, terminated, truncated))

        score += reward
        state = next_state

        if truncated or terminated:
            break

    if memory.size() > 2000:
        train(memory, V, V_target, V_optimizer)

        if episode % 10 == 0:
            soft_update(V_target, V)

    print(f"Episode : {episode} || Reward : {score} ")
    episode += 1

env.close()
