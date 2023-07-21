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

        self.fc1 = nn.Linear(N_STATE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)

        self.fc_q = nn.Linear(256, N_ATOM*N_ACTION)

        """
        
            return Q(state, left) , Q(state, right)        
            Q(state, left) -> 51개의 Atom의 Softmax 처리된 distribution의 형태  
                    
        """

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

N_ATOM   = 51
N_STATE  = 4
N_ACTION = 2
V_MAX    = 10
V_MIN    = -10
DELTA_ATOM = (V_MAX - V_MIN) / (N_ATOM - 1)
ATOM       = torch.FloatTensor(np.array([V_MIN + (i*DELTA_ATOM) for i in range(N_ATOM)])).to(device)


def Update_Q(buffer, Q, Q_target, Q_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = buffer.sample(batch_size)

    # Type casting & fit dimension
    terminateds = torch.unsqueeze(terminateds.type(torch.FloatTensor).to(device), dim=1)
    rewards = torch.unsqueeze(rewards, dim=1)

    # Make Value functions
    probs_   = Q(states)
    actions  = torch.squeeze(actions, dim = 1)

    probs_next_ = Q_target(next_states)  # 2 x 51
    Z_next = (probs_next_ * ATOM)  # Value distribution (with 51 atoms)
    Z_next_sum = torch.sum(Z_next, 2)
    next_optimal_action_idxes = torch.argmax(Z_next_sum, axis=1)  # 64,

    # target probs -> cross entropy loss
    m_prob     = torch.zeros((batch_size,N_ACTION, N_ATOM)).to(device) # 64 x 51
    probs      = torch.zeros((batch_size, N_ATOM)).to(device)
    probs_next = torch.zeros((batch_size, N_ATOM)).to(device)

    for i in range(batch_size):
        probs[i] = probs_[i][actions[i]]
        probs_next[i] = probs_next_[i][next_optimal_action_idxes[i]]

    for i in range(batch_size):
        for j in range(N_ATOM):
            Tz = min(V_MAX, max(V_MIN, rewards[i] + (gamma * ATOM[j]) * (1 - terminateds[i]))) # Tz = r + gamma*(Z) -> inducing disjoints of atoms
            bj = (Tz - V_MIN) / DELTA_ATOM
            m_l, m_u = math.floor(bj), math.ceil(bj)

            if bj != 50:
                m_prob[i][actions[i]][int(m_l)] += probs_next[i][j] * (m_u - bj.item())
                m_prob[i][actions[i]][int(m_u)] += probs_next[i][j] * (bj.item()  - m_l)

    m_prob_ = torch.zeros((batch_size, N_ATOM)).to(device)
    for i in range(batch_size):
        m_prob_[i] = m_prob[i][actions[i]]

    loss = loss_fc(m_prob_, probs)

    print(loss)
    Q_optimizer.zero_grad()
    loss.backward()
    Q_optimizer.step()

loss_fc = nn.CrossEntropyLoss()
Q = QNetwork().to(device)
Q_target = QNetwork().to(device)

Q_target.load_state_dict(Q.state_dict()) # Weight Synchronize
Q_optimizer = torch.optim.Adam(Q.parameters(), lr = 0.005) # Define Optimizer

MAX_EPISODE    = 1000
max_time_steps = 2000
reward_history = deque(maxlen = 1000)
ReplayBuffer = ReplayBuffer()
gamma = 0.99
batch_size = 32

env = gym.make("CartPole-v1") #

for episode in range(MAX_EPISODE):

    total_reward = 0 # total reward along the episode
    state = env.reset()
    state = torch.tensor(state[0]).float().to(device)

    for t in range(1, max_time_steps+1):
        if random.random() < 0.01:
            action = env.action_space.sample()
        else:
            # Make Value function
            probs = Q(state).to(device) # 2 x 51
            Z     = (probs * ATOM.to(device)) # Value distribution (with 51 atoms)
            Z_sum = torch.sum(Z, 1)      # (2, 51) x (51, ) -> (2, )
            action = torch.argmax(Z_sum, axis = 0).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = torch.tensor(next_state).float().to(device)
        total_reward += reward

        # Append to replay buffer
        ReplayBuffer.put([state, action, reward, next_state, terminated, truncated])

        # Update Q Network
        if ReplayBuffer.size() > 1000:
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