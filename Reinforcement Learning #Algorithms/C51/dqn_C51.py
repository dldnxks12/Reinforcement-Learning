"""

Test C51 based on DQN

*reference

- https://kminseo.tistory.com/16
- https://github.com/Kchu/DeepRL_PyTorch/blob/master/Distributional_RL/1_C51.py
- https://github.com/flyyufelix/C51-DDQN-Keras/blob/6921e1d8b702c2ff924b6861157d66fd7aa7c0e8/c51_ddqn.py#L149
- https://github.com/Kchu/DeepRL_PyTorch/blob/master/Distributional_RL/1_C51.py

"""

import sys
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque # for replay buffer

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"On {device}")

class QNetwork(nn.Module): # In state : 4 Out action : 2
    def __init__(self):
        super().__init__() # nn.Module의 constructor bringing

        self.fc1 = nn.Linear(N_STATE, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_q = nn.Linear(256, N_ATOM*N_ACTION)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        if x.shape == (batch_size, 256):
            q = self.fc_q(x).view(batch_size, N_ACTION, N_ATOM)
            q = F.softmax(q, dim = 2)  # 64 x 51 x 2

        else:
            q = self.fc_q(x).view(N_ACTION, N_ATOM)
            q = F.softmax(q, dim=1)  # 51 x 2

        return q


class ReplayBuffer():
    def __init__(self):
        super().__init__()
        self.buffer = deque(maxlen=20000)

    def put(self, sample):
        self.buffer.append(sample)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], []

        for state, action, reward, next_state, terminated, truncated in samples:
            states.append(state.cpu().numpy())
            actions.append([action])
            rewards.append(reward)
            next_states.append(next_state.cpu().numpy())
            terminateds.append(terminated)
            truncateds.append(truncated)
        states = torch.tensor(states, device=device)
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, device=device)
        next_states = torch.tensor(next_states, device=device)
        terminateds = torch.tensor(terminateds, device=device)
        truncateds = torch.tensor(truncateds, device=device)

        return states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)

MAX_EPISODE    = 1000
max_time_steps = 2000
reward_history = deque(maxlen = 1000)
ReplayBuffer = ReplayBuffer()
gamma = 0.99
batch_size = 32

N_ATOM   = 51
N_STATE  = 4
N_ACTION = 2
V_MAX    = 30
V_MIN    = -10
DELTA_ATOM = (V_MAX - V_MIN) / (N_ATOM - 1)
ATOM       = np.array([V_MIN + (i*DELTA_ATOM) for i in range(N_ATOM)])

def Update_Q(buffer, Q, Q_target, Q_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = buffer.sample(batch_size)

    # Type casting & fit dimension
    terminateds = torch.unsqueeze(terminateds.type(torch.FloatTensor).to(device), dim=1)
    rewards = torch.unsqueeze(rewards, dim=1)

    # Make Value functions
    probs_   = Q(states).detach().cpu().numpy()
    actions  = torch.squeeze(actions, dim = 1)

    probs_next_ = Q_target(next_states).detach().cpu().numpy()  # 2 x 51
    Z_next = np.multiply(probs_next_, ATOM)  # Value distribution (with 51 atoms)
    Z_next_sum = np.sum(Z_next, axis = 2)
    next_optimal_action_idxes = np.argmax(Z_next_sum, axis=1)  # 64,

    probs = np.zeros((batch_size, N_ATOM))
    probs_next = np.zeros((batch_size, N_ATOM))
    m_prob = np.zeros((batch_size, N_ACTION, N_ATOM))  # 64 x 51

    # target probs -> cross entropy loss
    for i in range(batch_size):
        probs[i] = probs_[i][actions[i]]
        probs_next[i] = probs_next_[i][next_optimal_action_idxes[i]]

    for i in range(batch_size):
        if terminateds[i]:
            Tz = min(V_MAX, max(V_MIN, rewards[i])) # Tz = r + gamma*(Z) -> inducing disjoints of atoms
            bj = (Tz - V_MIN) / DELTA_ATOM
            m_l, m_u = math.floor(bj), math.ceil(bj)
            m_prob[i][actions[i]][int(m_l)] += (m_u - bj)
            m_prob[i][actions[i]][int(m_u)] += (bj  - m_l)
        else:
            for j in range(N_ATOM):
                Tz = min(V_MAX, max(V_MIN, rewards[i] + (gamma * ATOM[j]))) # Tz = r + gamma*(Z) -> inducing disjoints of atoms
                bj = (Tz - V_MIN) / DELTA_ATOM
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[i][actions[i]][int(m_l)] += probs_next[i][j] * (m_u - bj)
                m_prob[i][actions[i]][int(m_u)] += probs_next[i][j] * (bj  - m_l)

    m_prob_ = np.zeros((batch_size, N_ATOM))
    for i in range(batch_size):
        m_prob_[i] = m_prob[i][actions[i]]

    m_prob_, probs = torch.FloatTensor(m_prob_).to(device), torch.FloatTensor(probs).to(device)
    loss = loss_fc(m_prob_.requires_grad_(True), probs.requires_grad_(True))
    print(loss)

    #loss = target_distribution * (-torch.log(probs + 1e-8))
    #loss = torch.mean(loss).requires_grad_(True)

    Q_optimizer.zero_grad()
    loss.backward()
    Q_optimizer.step()

loss_fc = nn.CrossEntropyLoss()
Q = QNetwork().to(device)
Q_target = QNetwork().to(device)

Q_target.load_state_dict(Q.state_dict()) # Weight Synchronize
Q_optimizer = torch.optim.Adam(Q.parameters(), lr = 0.001) # Define Optimizer

env = gym.make("CartPole-v1") #, render_mode = "human")

for episode in range(MAX_EPISODE):

      total_reward = 0 # total reward along the episode
      state = env.reset()
      state = torch.tensor(state[0]).float().to(device)

      for t in range(1, max_time_steps+1):
            with torch.no_grad():
                if random.random() < 0.01:
                    action = env.action_space.sample()
                else:
                    # Make Value function
                    probs = Q(state).cpu().numpy() # 2 x 51
                    Z     = np.multiply(probs, ATOM) # Value distribution (with 51 atoms)
                    Z_sum = np.sum(Z, axis = 1)      # (2, 51) x (51, ) -> (2, )
                    action = np.argmax(Z_sum, axis = 0)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(next_state).float().to(device)
            total_reward += reward

            # Append to replay buffer
            ReplayBuffer.put([state, action, reward, next_state, terminated, truncated])

            # Update Q Network
            if ReplayBuffer.size() > 100:
                  Update_Q(ReplayBuffer, Q, Q_target, Q_optimizer)

            # Periodic Update
            if max_time_steps % 10 == 0:
                Q_target.load_state_dict(Q.state_dict())

            if terminated or truncated:
                  break

            state = next_state

      reward_history.append(total_reward)
      avg = sum(reward_history) / len(reward_history)
      print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, total_reward, avg))

env.close()