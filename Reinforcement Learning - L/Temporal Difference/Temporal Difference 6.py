from collections import deque
import os
import sys
import gym
import random
import numpy as np

# torchs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Currently on {device}")

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=10000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n_samples):
        mini_batch = random.sample(self.buffer, n_samples)
        states, actions, rewards, next_states, terminateds, truncateds = [],[],[],[],[],[]

        for state, action, reward, next_state, terminated, truncated in mini_batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)
            truncateds.append(truncated)

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1    = nn.Linear(4, 48)
        self.fc2    = nn.Linear(48, 64)
        self.fc_out = nn.Linear(64, 2) # Left or Right

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)

        return x

def Update_Q(Buffer, Q, Q_target, batch_size, Q_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = Buffer.sample(batch_size)

    loss = 0
    for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):
        if terminated or truncated:
            y = reward
        else:
            y = reward + gamma*max(Q_target(next_state))

        loss += (y - Q(state)[action])**2

    loss = (loss/batch_size) # Gradient Descent

    Q_optimizer.zero_grad()
    loss.backward()
    Q_optimizer.step()


e  = 0.2
lr = 0.0005
tau = 0.05
gamma = 0.99

Q = QNetwork().to(device)
Q_optimizer = optim.Adam(Q.parameters(), lr = lr)

Q_target = QNetwork().to(device)
Q_target.load_state_dict(Q.state_dict()) # Synchronize Q, Q_target parameters

Buffer = ReplayBuffer()
env = gym.make('CartPole-v0')

max_time_step = 1000
MAX_EPISODE = 200
for episode in range(MAX_EPISODE):

    observation = env.reset()[0]
    state       = torch.tensor(observation).to(device)
    terminated, truncated = False, False
    total_reward = 0

    for time_step in range(max_time_step):
        with torch.no_grad():
            if random.random() < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q(state).cpu()).item()

        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(next_observation).to(device)

        Buffer.put([state, action, reward, next_state, terminated, truncated])
        total_reward += reward

        if Buffer.size() > 500:
            Update_Q(Buffer, Q, Q_target, 32, Q_optimizer)

            if time_step % 50 == 0:
                Q_target.load_state_dict(Q.state_dict())

        if terminated or truncated:
            break

        observation = next_observation
        state = next_state

    print(f"Episode : {episode} | Total_reward : {total_reward}")

env.close()

#test
env_test = gym.make('CartPole-v0', render_mode = 'human')
episode = 0

state = env_test.reset()[0]
while episode < 10:
    state = torch.tensor(state).to(device)
    action = torch.argmax(Q(state)).item()
    next_state, reward, terminated, truncated, info = env_test.step(action)
    state = next_state

    if terminated or truncated:
        state = env_test.reset()

env_test.close()


