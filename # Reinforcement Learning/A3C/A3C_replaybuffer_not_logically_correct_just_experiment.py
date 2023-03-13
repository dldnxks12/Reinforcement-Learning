"""

My first distributed reinforcement learning - A3C

*reference1 : https://github.com/keep9oing/PG-Family/blob/main/A3C.py
*reference2 : https://github.com/seungeunrho/minimalRL/blob/master/a3c.py

"""

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
        dones = torch.tensor(dones)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

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

def seed_torch(seed):
    torch.manual_seed(seed) # seed 고정
    if torch.backends.cudnn.enabled == True:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

beta = 0.01
def soft_update(Target_Network, Current_Network):
    for target_param, current_param in zip(Target_Network.parameters(), Current_Network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - beta) + current_param.data * beta)


def train(global_model_pi, global_model_v, rank, device):

    # Fixing seed for reducibility
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    seed_torch(seed + rank)

    local_model_pi           = PolicyNetwork()
    local_model_v            = ValueNetwork()
    local_model_v_target = ValueNetwork()

    local_model_pi.load_state_dict(global_model_pi.state_dict())
    local_model_v.load_state_dict(global_model_v.state_dict())
    local_model_v_target.load_state_dict(global_model_v.state_dict())

    optimizer_pi = optim.Adam(global_model_pi.parameters(), lr = 0.0002)
    optimizer_v  = optim.Adam(global_model_v.parameters(), lr=0.0002)

    memory = ReplayBuffer()
    env = gym.make("CartPole-v1")
    update_interval = 5
    for e in range(episodes):
        state = env.reset()[0]
        done = False
        while True:
            prob = local_model_pi(torch.from_numpy(state).float())
            m = Categorical(prob)
            action = m.sample().item()
            next_state, reward, done, _, _ = env.step(action)
            memory.put((state, action, reward/100.0, next_state, done))

            state = next_state

            if done:
                break

            if memory.size() > 50:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                # Critic Update
                critic_loss = 0
                for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                    if d:
                        y = r
                    else:
                        y = r + gamma * local_model_v_target(ns)

                    critic_loss += (y - local_model_v(s)) ** 2

                critic_loss /= batch_size
                optimizer_v.zero_grad()
                for global_param, local_param in zip(global_model_v.parameters(), local_model_v.parameters()):
                    global_param._grad = local_param.grad
                optimizer_v.step()
                local_model_v.load_state_dict(global_model_v.state_dict())


                # Actor update
                actor_loss = 0
                for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                    actor_loss += (r + (gamma * local_model_v(ns)) - local_model_v(s)) * ((local_model_pi(s)[a] + 1e-5).log())

                actor_loss = -actor_loss/batch_size

                optimizer_pi.zero_grad()
                actor_loss.backward()
                for global_param, local_param in zip(global_model_pi.parameters(), local_model_pi.parameters()):
                    global_param._grad = local_param.grad
                optimizer_pi.step()
                local_model_pi.load_state_dict(global_model_pi.state_dict())

                soft_update(local_model_v_target, local_model_v)

    print(f"# ------------------------ P {rank} ends ----------------------------- #")
    env.close()

def test(global_model_pi, global_model_v, device):
    env = gym.make('CartPole-v1')
    score = 0.0
    print_interval = 20

    for n_epi in range(5000):
        done = False
        s = env.reset()[0]
        while not done:
            prob = global_model_pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().item()
            s_prime, r, done, _, _ = env.step(a)
            s = s_prime
            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
            time.sleep(1)
    env.close()

actor_lr = 0.0002
critic_lr = 0.0002
episodes = 300
gamma = 0.98
batch_size = 32
seed = 100

if __name__ == "__main__":
    # set gym environment

    if torch.cuda.is_available():
        device = torch.device("cuda")

    device = 'cpu'  # Work well
    #device = 'cuda' # Work bad

    # Seed fixing
    seed_torch(seed)     # torch seed fix
    random.seed(seed)    # random seed fix
    np.random.seed(seed) # numpy seed fix

    processes   = []
    process_num = 8 # Number of threads

    """
        CUDA는 start method로 오직 spawn, forkserver만 제공한다.        
        -> multi-threaded 프로그램을 fork하는 순간 child process가 무조건 죽어버리는 현재의 os 디자인이 원인
    """
    mp.set_start_method('spawn')
    print("MP start method : ", mp.get_start_method())

    global_model_pi           = PolicyNetwork()
    global_model_v        = ValueNetwork()

    global_model_pi.share_memory()
    global_model_v.share_memory()

    processes = []
    for rank in range(process_num):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(global_model_pi, global_model_v, device))
        else:
            p = mp.Process(target=train, args=(global_model_pi, global_model_v, rank, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()














