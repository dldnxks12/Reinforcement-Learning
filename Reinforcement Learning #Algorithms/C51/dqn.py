from collections import deque # for replay buffer

import sys # for debugging
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Currently working on {device}")

class QNetwork(nn.Module):
      def __init__(self):
            super().__init__()  # nn.Module의 constructor bringing

            self.fc1 = nn.Linear(4, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 256)

            self.fc_q = nn.Linear(256, 2)

      def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            q = self.fc_q(x)

            return q


class ReplayBuffer_():
      def __init__(self):
            self.buffer = deque(maxlen = 10000)

      def put(self, transition):
            self.buffer.append(transition)

      def sample(self, n):
            mini_batch = random.sample(self.buffer, n)
            states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], []

            for transition in mini_batch:
                  state, action, reward, next_state, terminated, truncated = transition

                  """
                  print(state, type(state))           # Tensor
                  print(action, type(action))         # int
                  print(reward, type(reward))         # float
                  print(next_state, type(next_state)) # Tensor
                  print(terminated, type(terminated)) # bool
                  print(truncated, type(truncated))   # bool 
                  """

                  states.append(state)
                  actions.append(action)
                  rewards.append(reward)
                  next_states.append(next_state)
                  terminateds.append(terminated)
                  truncateds.append(truncated)

            return states, actions, rewards, next_states, terminateds, truncateds

      def size(self):
            return len(self.buffer)

Q = QNetwork().to(device)
Q_target = QNetwork().to(device)

Q_target.load_state_dict(Q.state_dict()) # Weight Synchronize
Q_optimizer = torch.optim.Adam(Q.parameters(), lr = 0.001) # Define Optimizer

max_time_steps = 2000
reward_history = deque(maxlen = 1000)
ReplayBuffer = ReplayBuffer_()
gamma = 0.99

def Update_Q(buffer, Q, Q_target, Q_optimizer):
      states, actions, rewards, next_states, terminateds, truncateds = buffer.sample(batch_size)

      loss = 0
      for state, action, reward, next_state, terminated, truncated in zip(states, actions, rewards, next_states, terminateds, truncateds):

            """
            print(state, type(state))           # Tensor
            print(action, type(action))         # int
            print(reward, type(reward))         # float
            print(next_state, type(next_state)) # Tensor 
            print(terminated, type(terminated)) # bool
            print(truncated, type(truncated))   # bool 
            """

            # Just inference, no weight update
            if (terminated == True) or (truncated == True):
                  y = reward
            else:
                  y = reward + ( gamma * max(Q_target(next_state)) )

            action = int(action)
            loss += (y - Q(state)[action])**2

      loss = loss / batch_size

      print(loss)
      Q_optimizer.zero_grad()
      loss.backward()
      Q_optimizer.step()

env = gym.make("CartPole-v1") #, render_mode = "human")
batch_size = 32

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
            ReplayBuffer.put([state, action, reward, next_state, terminated, truncated])

            # Update Q Network
            if ReplayBuffer.size() > 1000:
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




