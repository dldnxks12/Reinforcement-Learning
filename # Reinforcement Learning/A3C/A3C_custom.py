"""

My first distributed reinforcement learning - A3C

*reference : https://github.com/keep9oing/PG-Family/blob/main/A3C.py

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

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

def seed_torch(seed):
    torch.manual_seed(seed) # seed 고정
    if torch.backends.cudnn.enabled == True: # cudnn 사용할 시...

        # network의 입력 사이즈가 달라지지 않을 때 사용하면 좋다.
        # 그 크기에 맞는 최적의 연산 알고리즘을 골라주기 때문
        # 따라서 입력 사이즈가 바뀌면 그때마다 최적 알고리즘을 찾기 때문에 연산 효율 떨어진다.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def train(global_model, rank):

    # Fixing seed for reducibility
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    seed_torch(seed + rank)

    local_model  = ActorCritic().to(device)
    local_model.load_state_dict(global_model.state_dict())
    global_optimizer  = optim.Adam(global_model.parameters(), lr = 0.0002)

    env = gym.make("CartPole-v1")
    update_interval = 5
    for e in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            states, actions, rewards = [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(state).float().to(device))
                m = Categorical(prob)
                action = m.sample().item()

                next_state, reward, done, _, _ = env.step(action)

                states.append(state)
                actions.append([action])
                rewards.append(reward/100.0)

                state = next_state

                if done:
                    break

            s_final = torch.tensor(next_state, dtype = torch.float).to(device)
            R = 0.0 if done else local_model.v(s_final).item()

            td_target_list = []
            for reward in rewards[::-1]: # 뒤에서부터 하나씩 -> N_step reward
                R = (gamma * R) + reward
                td_target_list.append([R])
            td_target_list.reverse()

            states = torch.tensor(states, dtype = torch.float).to(device)
            actions = torch.tensor(actions).to(device)
            td_targets = torch.tensor(td_target_list).to(device)

            advantage = (td_targets - local_model.v(states))

            pi = local_model.pi(states, softmax_dim=1)
            pi_a = pi.gather(1, actions)

            # Critic loss + Actor loss
            loss = advantage.detach() * (-torch.log(pi_a)) + F.smooth_l1_loss(local_model.v(states), td_targets.detach())
            global_optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            global_optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
    print(f"# ------------------------ P {rank} ends ----------------------------- #")
    env.close()

def test(global_model):
    env = gym.make('CartPole-v1')
    score = 0.0
    print_interval = 20

    for n_epi in range(400):
        done = False
        s = env.reset()[0]
        while not done:
            prob = global_model.pi(torch.from_numpy(s).float().to(device))
            a = Categorical(prob).sample().item()
            s_prime, r, done, _, _ = env.step(a)
            s = s_prime
            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score/print_interval))
            score = 0.0
            time.sleep(1)
    env.close()

actor_lr = 0.0002
critic_lr = 0.0002
episodes = 10000
gamma = 0.98
batch_size = 5
seed = 100

hidden_layer_num = 2
hidden_dim_size = 128

if __name__ == "__main__":
    # set gym environment

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Seed fixing
    seed_torch(seed)     # torch seed fix
    random.seed(seed)    # random seed fix
    np.random.seed(seed) # numpy seed fix


    processes   = []
    process_num = 3 # Number of threads

    """
        CUDA는 start method로 오직 spawn, forkserver만 제공한다.        
        -> multi-threaded 프로그램을 fork하는 순간 child process가 무조건 죽어버리는 현재의 os 디자인이 원인
    """
    mp.set_start_method('spawn')
    print("MP start method : ", mp.get_start_method())

    global_model = ActorCritic().to(device)
    global_model.share_memory()

    processes = []
    for rank in range(process_num):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(global_model,))
        else:
            p = mp.Process(target=train, args=(global_model, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()















