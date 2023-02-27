"""

Test C51 based on DQN

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

        if x.shape == (64, 256):
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
DELTA_ATOM = (V_MAX - V_MIN) / (float(N_ATOM) - 1)
ATOM       = np.array([V_MIN + (i*DELTA_ATOM) for i in range(N_ATOM)])

def Update_Q(buffer, Q, Q_target, Q_optimizer):
    states, actions, rewards, next_states, terminateds, truncateds = buffer.sample(batch_size)

    # Type casting & fit dimension
    terminateds = torch.unsqueeze(terminateds.type(torch.FloatTensor).to(device), dim=1)
    rewards = torch.unsqueeze(rewards, dim=1)

    loss = 0

    m_prob = [np.zeros((batch_size, N_ATOM)) for i in range(N_ACTION)]
    print(np.array(m_prob).shape)
    sys.exit()
    z  = Q(next_states).detach().cpu().numpy() # 64 x 2 x 51
    z_ = Q_target(next_states).detach().cpu().numpy()

    q_eval = np.sum(np.multiply(z, ATOM), axis = 2)
    optimal_action_idxes = np.argmax(q_eval, axis = 1) # 64,

    # Project next state value distribution of optimal action to current state

    for i in range(batch_size):
        if terminateds[i]: # if terminates == 1 -> end of episode
            Tz = min(V_MAX, max(V_MIN, rewards[i]))
            bj = (Tz - V_MIN) / DELTA_ATOM
            m_l, m_u = math.floor(bj), math.ceil(bj)
            m_prob[action[i]][i][int(m_l)] += (m_u - bj)
            m_prob[action[i]][i][int(m_u)] += (bj - m_l)
        else:
            for j in range(N_ATOM):
                Tz = min(V_MAX, max(V_MIN, rewards[i] + gamma*z[j]))
                bj = (Tz - V_MIN) / DELTA_ATOM
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxes[i]][i][j] * (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxes[i]][i][j] * (bj - m_l)


    Q_optimizer.zero_grad()
    loss.backward()
    Q_optimizer.step()

Q = QNetwork().to(device)
Q_target = QNetwork().to(device)

Q_target.load_state_dict(Q.state_dict()) # Weight Synchronize
Q_optimizer = torch.optim.Adam(Q.parameters(), lr = 0.001) # Define Optimizer

MAX_EPISODE    = 1000
max_time_steps = 2000
reward_history = deque(maxlen = 1000)
ReplayBuffer = ReplayBuffer()
gamma = 0.99
batch_size = 64

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
                    z = Q(state).cpu().numpy() # 2 x 51
                    q = np.sum(np.multiply(z, ATOM), axis = 1) # (2, 51) x (51, ) -> (2, )
                    action = np.argmax(q, axis = 0)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(next_state).float().to(device)
            total_reward += reward

            # Append to replay buffer
            ReplayBuffer.put([state, action, reward, next_state, terminated, truncated])

            # Update Q Network
            if ReplayBuffer.size() > 100:
                  Update_Q(ReplayBuffer, Q, Q_target, Q_optimizer)

            if terminated or truncated:
                  break

            state = next_state

      # Periodic Update
      if episode % 10 == 0:
            Q_target.load_state_dict(Q.state_dict())

      reward_history.append(total_reward)
      avg = sum(reward_history) / len(reward_history)
      print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, total_reward, avg))

env.close()