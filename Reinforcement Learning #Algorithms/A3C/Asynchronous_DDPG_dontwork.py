import sys
import time
import random
import numpy as np
import gym
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque, namedtuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen = 50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, number):
        mini_batch = random.sample(self.buffer, number)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done = transition

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states      = torch.tensor(states)
        actions     = torch.tensor(actions)
        rewards     = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        dones = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

# Actor
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)   # Input  : state
        self.fc2 = nn.Linear(64, 64)  # Input  : state
        self.fc3 = nn.Linear(64, 1)   # Output : action

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        action = torch.tanh(self.fc3(state)) * 2

        return action

# Critic
class QNetwork(nn.Module): # Q function approximation
    def __init__(self):
        super().__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, state, action):
        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))

        concatenate = torch.cat([h1, h2], dim = -1)

        Q = F.relu(self.fc1(concatenate))
        Q = F.relu(self.fc2(Q))

        return self.out(Q)

# Add noise to Action
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

def seed_torch(seed):
    torch.manual_seed(seed) # seed 고정
    if torch.backends.cudnn.enabled == True:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def soft_update(Target_Network, Current_Network):
    for target_param, current_param in zip(Target_Network.parameters(), Current_Network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - beta) + current_param.data * beta)

def train(global_model_pi, global_model_q, rank, device):
    # Fixing seed for reducibility
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    seed_torch(seed + rank)

    local_model_pi           = PolicyNetwork()
    local_model_pi_target    = PolicyNetwork()
    local_model_q            = QNetwork()
    local_model_q_target     = QNetwork()

    local_model_pi.load_state_dict(global_model_pi.state_dict())
    local_model_pi_target.load_state_dict(global_model_pi.state_dict())

    local_model_q.load_state_dict(global_model_q.state_dict())
    local_model_q_target.load_state_dict(global_model_q.state_dict())

    optimizer_pi = optim.Adam(global_model_pi.parameters(), lr = actor_lr)
    optimizer_q  = optim.Adam(global_model_q.parameters(), lr = critic_lr)

    memory = ReplayBuffer()
    noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    env = gym.make('Pendulum-v1')                          # Rendering = Off

    for e in range(episodes):
        print(f"Current process : {rank} | Episode : {e}")
        state = env.reset()[0]
        done = False
        for step in range(MAX_STEP):
            with torch.no_grad():
                action = local_model_pi(torch.from_numpy(state)) + noise()[0]
            next_state, reward, done, _, _ = env.step(action.cpu().detach().numpy())

            if step == MAX_STEP-1:
                done = True
            memory.put((state, action, reward, next_state, done))

            state = next_state

            if done:
                break

            if memory.size() > 50:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                terminateds = torch.unsqueeze(dones, dim=1)
                rewards = torch.unsqueeze(rewards, dim=1)
                actions = torch.unsqueeze(actions, dim=1)

                # Critic Update
                critic_loss = 0
                with torch.no_grad():
                    acts = local_model_pi_target(next_states)
                    y = rewards + gamma * local_model_q_target(next_states, acts) * (1 - dones)

                critic_loss = ((y - local_model_q(states, actions)) ** 2).mean()

                optimizer_q.zero_grad()
                for global_param, local_param in zip(global_model_q.parameters(), local_model_q.parameters()):
                    global_param._grad = local_param.grad
                optimizer_q.step()
                local_model_q.load_state_dict(global_model_q.state_dict())

                # Actor update
                actor_loss = 0

                actor_loss = local_model_q(states, local_model_pi(states)).mean() * (-1)

                for p in local_model_q.parameters():
                    p.requires_grads = False

                optimizer_pi.zero_grad()
                actor_loss.backward()
                for global_param, local_param in zip(global_model_pi.parameters(), local_model_pi.parameters()):
                    global_param._grad = local_param.grad
                optimizer_pi.step()
                local_model_pi.load_state_dict(global_model_pi.state_dict())

                for p in local_model_q.parameters():
                    p.requires_grads = True

                soft_update(local_model_q_target, local_model_q)
                soft_update(local_model_pi_target, local_model_pi)

    print(f"# ------------------------ P {rank} ends ----------------------------- #")
    env.close()


def test(global_model_pi , device):
    env = gym.make('Pendulum-v1')
    score = 0.0
    print_interval = 20
    noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    for n_epi in range(5000):
        done = False
        s = env.reset()[0]
        for step in range(MAX_STEP):
            a = global_model_pi(torch.from_numpy(s).float())
            s_prime, r, done, _, _ = env.step(a.cpu().detach().numpy())
            s = s_prime
            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
            time.sleep(1)
    env.close()

actor_lr  = 0.0001
critic_lr = 0.001
episodes = 300
gamma = 0.98
beta = 0.001
batch_size = 32
seed = 100
MAX_STEP = 1000

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")

    device = 'cpu'  # Work well

    # Seed fixing
    seed_torch(seed)     # torch seed fix
    random.seed(seed)    # random seed fix
    np.random.seed(seed) # numpy seed fix

    processes   = []
    process_num = 5 # Number of threads

    mp.set_start_method('spawn')
    print("MP start method : ", mp.get_start_method())

    global_model_pi   = PolicyNetwork()
    global_model_q    = QNetwork()

    global_model_pi.share_memory()
    global_model_q.share_memory()

    processes = []
    for rank in range(process_num + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(global_model_pi, device))
        else:
            p = mp.Process(target=train, args=(global_model_pi, global_model_q, rank, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()














