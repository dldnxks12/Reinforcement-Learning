# Monte Carlo Policy Gradient Method

import sys
import gym
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Policy Network
class REINFORCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 24) # Input : state
        self.fc2 = nn.Linear(24, 2)

    def forward(self, state):
        state  = self.fc1(state)
        state  = F.relu(state)
        state  = self.fc2(state)
        action = F.softmax(state, dim = -1)

        return action # Softmax result (Policy는 전체 action의 합이 1이 되어야하므로 ...)

gamma = 0.99
alpha = 0.002

def G(rewards):
    G_0 = 0
    for i in range(len(rewards)-1):
        gam = math.pow(gamma, i) # gamma의 i제곱
        G_0 += gam*rewards[i]
    return G_0

def Gen_Episode(ep):
    states, actions, rewards = [], [], []

    state = env.reset()
    state = torch.tensor(state[0]).float().to(device)

    terminated = False
    truncated  = False
    score = 0

    while True:
        probabilities = pi(state) # policy에 대한 Softmax 확률 return
        action = torch.multinomial(probabilities, 1).item() # Index return

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = torch.tensor(next_state).float().to(device)

        if terminated or truncated:
            reward = -10

        score += reward

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        if terminated or truncated:
            break

    return states, actions, rewards, score

pi           = REINFORCE().to(device) # Policy Network
pi_optimizer = optim.Adam(pi.parameters(), lr = alpha)

env = gym.make("CartPole-v1")
episode     = 0
MAX_EPISODE = 100000

while episode < MAX_EPISODE:
    states, actions, rewards, score = Gen_Episode(episode)

    # Get G
    G_0 = G(rewards)

    loss_temp = 0
    for state, action in zip(states, actions):

        soft_max_action = pi(state)
        log_pi          = soft_max_action[action].log() # log_pi[a|s]
        loss_temp       += log_pi

    loss = -loss_temp * G_0
    pi_optimizer.zero_grad()
    loss.backward()
    pi_optimizer.step()

    print(f"Episode : {episode} || Rewards : {score} || Loss : {loss.item()}")
    episode += 1

env.close()

