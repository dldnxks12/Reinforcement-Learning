"""

Play Cartpole game with Dueling Double Deep Q Network


* reference

    https://sezan92.github.io/2020/03/18/D3QN.html

"""

import gym
import sys
import math
import random
import random
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from time import sleep
from collections import deque


class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=20000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n_samples):
        mini_batch = random.sample(self.buffer, n_samples)

        states, actions, rewards, next_states, terminateds, truncateds = [] ,[] ,[] ,[] ,[] ,[]

        for state, action, reward, next_state, terminated, truncated in mini_batch:
            states.append(state.cpu().numpy())
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.cpu().numpy())
            terminateds.append(terminated)
            truncateds.append(truncated)

        states = torch.tensor(states, device=device)
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, device=device)
        next_states = torch.tensor(next_states, device=device)
        terminateds = torch.tensor(terminateds, device=device, dtype = torch.float)
        truncateds = torch.tensor(truncateds, device=device, dtype = torch.float)

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)

        # Value function layer
        self.v1    = nn.Linear(64, 64)
        self.v2    = nn.Linear(64, 32)
        self.v_out = nn.Linear(32, 1)

        # Advantage function layer
        self.a1    = nn.Linear(64, 64)
        self.a2    = nn.Linear(64, 32)
        self.a_out = nn.Linear(32, 2) # output : action size

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        v = F.relu(self.v1(x))
        v = F.relu(self.v2(v))
        v = self.v_out(v) # V

        a = F.relu(self.a1(x))
        a = F.relu(self.a2(a))
        a = self.a_out(a) # A(, 1), A(, 2)

        Q = v + a - a.mean()

        return Q, v, a

def soft_update(Agent, Agent_target):
   for target_param, param in zip(Agent_target.parameters(), Agent.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

def train(Buffer, Agent, Agent_target, Agent_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = Buffer.sample(Batch_size)

    actions = torch.unsqueeze(actions, dim = 1)
    rewards = torch.unsqueeze(rewards, dim = 1)
    terminateds = torch.unsqueeze(terminateds, dim = 1)

    loss = 0

    with torch.no_grad():
        Q_target, v_target, a_target = Agent_target(next_states)

    Q_next, v_next, a_next  = Agent(next_states)
    Q_cur, v_cur, a_cur     = Agent(states) # for calc loss

    predicted_action = torch.unsqueeze(torch.argmax(Q_next, dim = 1), dim = 1)
    y = rewards + (gamma * Q_target.gather(1, predicted_action) * (1 - terminateds))

    # Calc gradient
    for p in Agent_target.parameters():
        p.requires_grad = False

    y_hat = Q_cur.gather(1, actions)
    loss = ( (y - y_hat)**2 ).mean()

    Agent_optimizer.zero_grad()
    loss.backward()
    Agent_optimizer.step()

    for p in Agent_target.parameters():
        p.requires_grad = True

    soft_update(Agent, Agent_target)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("")
print(f"# - Currently on {device} - #")
print("")

env = gym.make("CartPole-v1", render_mode = 'human')
#env = gym.make("CartPole-v1")
Buffer = ReplayBuffer()
MAX_EPISODE = 2000
MAX_STEP    = 1000
Batch_size  = 64

gamma = 0.9
tau   = 0.005
lr    = 0.001

agent        = Network().to(device)
agent_target = Network().to(device)
agent_optimizer = optim.Adam(agent.parameters(), lr = lr)

agent_target.load_state_dict(agent.state_dict()) # parameter sync

for episode in range(MAX_EPISODE):

    state, _ = env.reset()
    state    = torch.tensor(state).to(device)

    total_reward = 0
    terminated   = False
    truncated    = False

    for step in range(MAX_STEP):

        with torch.no_grad():
            Q, v, a = agent(state) # forward ...

            # Soft greedy policy
            if random.random() < 0.1:
                action = env.action_space.sample()
            else:
                action = torch.argmax(Q).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(next_state).to(device)

        Buffer.put([state, action, reward, next_state, terminated, truncated])
        total_reward += reward

        if Buffer.size() > 100:
            train(Buffer, agent, agent_target, agent_optimizer)

        if terminated or truncated:
            break

        state = next_state

    print(f"Episode : {episode} || total reward : {total_reward}")

env.close()