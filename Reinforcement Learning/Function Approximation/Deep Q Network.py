from datetime import datetime
from collections import deque # for replay buffer

from torch.distributions import Categorical

import os
import sys # for debugging
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Currently working on {device}")

class QNetwork(nn.Module):
      def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 48) # Input : 4개의 state
            self.fcQ1 = nn.Linear(48, 64)
            self.fcQ2 = nn.Linear(64, 2) # Output : 2개 Action - left/right

      def forward(self, states):
            states  = self.fc(states)
            states  = F.relu(states)
            states  = self.fcQ1(states)
            states  = F.relu(states)
            actions = self.fcQ2(states)

            return actions

class ReplayBuffer_():
      def __init__(self):
            self.buffer = deque(maxlen = 50000)

      def put(self, transition):
            self.buffer.append(transition)

      def sample(self, n):
            mini_batch = random.sample(self.buffer, n)
            states, actions, rewards, next_states, terminateds = [], [], [], [], []

            for transition in mini_batch:
                  state, action, reward, next_state, terminated = transition

                  """
                  print(state, type(state))           # Tensor
                  print(action, type(action))         # int
                  print(reward, type(reward))         # float
                  print(next_state, type(next_state)) # Tensor
                  print(terminated, type(terminated)) # bool
                  """

                  states.append(state)
                  actions.append(action)
                  rewards.append(reward)
                  next_states.append(next_state)
                  terminateds.append(terminated)

            return states, actions, rewards, next_states, terminateds

      def size(self):
            return len(self.buffer)

Q = QNetwork().to(device)
Q_target = QNetwork().to(device)

max_time_steps = 1000
reward_history = deque(maxlen = 1000)
ReplayBuffer = ReplayBuffer_()
gamma = 0.99

Q_optimizer = torch.optim.Adam(Q.parameters(), lr = 0.001)

def Update_Q(buffer, Q, Q_target, Q_optimizer):
      states, actions, rewards, next_states, terminateds = buffer.sample(128)

      loss = 0
      for state, action, reward, next_state, terminated in zip(states, actions, rewards, next_states, terminateds):

            """
            print(state, type(state))           # Tensor
            print(action, type(action))         # int
            print(reward, type(reward))         # float
            print(next_state, type(next_state)) # Tensor
            print(terminated, type(terminated)) # bool
            """

            if terminated == 0:
                  y = reward
            else:
                  y = reward * gamma*max(Q_target(next_state))

            action = int(action)
            loss += (y - Q(state)[action])**2

      loss = loss/128

      Q_optimizer.zero_grad()
      loss.backward()
      Q_optimizer.step()

env = gym.make("CartPole-v0", render_mode = "human")

for episode in range(10000):

      total_reward = 0 # total reward along the episode
      state = env.reset()
      state = torch.tensor(state[0]).float().to(device)

      for t in range(1, max_time_steps+1):
            with torch.no_grad():
                  if random.random() < 0.01:
                        action = env.action_space.sample()
                  else:
                        action = torch.argmax(Q(state)).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(next_state).float().to(device)

            total_reward += reward

            # Append to replay buffer
            ReplayBuffer.put([state, action, reward, next_state, terminated])

            # Update Q Network
            if ReplayBuffer.size() > 1000:
                  Update_Q(ReplayBuffer, Q, Q_target, Q_optimizer)

                  # Periodic Update
                  if episode % 10 == 0:
                        Q_target.load_state_dict(Q.state_dict())

            if terminated or truncated:
                  break

            state = next_state

      reward_history.append(total_reward)
      avg = sum(reward_history) / len(reward_history)
      print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, total_reward, avg))

env.close()




