import os
import random
import gymnasium as gym
import itertools
import argparse
import numpy as np
import torch
from sac import SAC
from buffer import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

parser.add_argument('--env-name', default="HalfCheetah-v4",
                    help='Mujoco Gym environment (default: HalfCheetah-v4)')

parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')

parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')

parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')

parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')

parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')

parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')

parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')

parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')

parser.add_argument('--num_steps', type=int, default=10000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')

parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')

parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')


args = parser.parse_args()

# Custom Environment (Inverted Single Pendulum)
xml_file = os.getcwd() + "/environment/assets/inverted_single_pendulum.xml"
env      = gym.make("InvertedSinglePendulum-v4", model_path=xml_file)
env_vis  = gym.make("InvertedSinglePendulum-v4", render_mode = 'human', model_path = xml_file)

#env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# delay sample
d_sample   = 9
s_time     = "50ms"
delay_args = [d_sample, s_time]
num_model  = 7

# Agent
agent = SAC(env.observation_space.shape[0] + d_sample, env.action_space, delay_args, num_model, args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates        = 0

for i_episode in itertools.count(1): # 1씩 증가시키는 무한 반복자

    episode_reward = 0
    episode_steps  = 0

    done = False
    state = env.reset()[0]
    state = torch.FloatTensor(state)

    act_buf = []
    for _ in range(d_sample):
        act_buf.append(torch.FloatTensor([0]))

    max_idx_box = []
    while not done:

        # augmented state space
        I = torch.concatenate([state.unsqueeze(0), torch.FloatTensor(act_buf[-d_sample:]).unsqueeze(0)], dim = 1)

        # TODO : Selecting action, according to voting ...
        if args.start_steps > total_numsteps:    # start_steps = default : 10000
            action = env.action_space.sample()   # Sample random action
            next_state, reward, terminated, truncated, _ = env.step(act_buf[-d_sample].numpy())  # Step
            next_state = torch.FloatTensor(next_state)
            action_idx = random.randint(0, num_model - 1)
            max_idx_box.append(action_idx)

        else:
            action, action_idx = agent.select_action(I, total_numsteps)  # Sample action from policy
            next_state, reward, terminated, truncated, _ = env.step(act_buf[-d_sample].numpy())  # Step
            next_state = torch.FloatTensor(next_state)
            max_idx_box.append(action_idx)

        # TODO : idx_box = most selected actors
        if len(memory) > args.batch_size:
            max_idx = max(max_idx_box, key = max_idx_box.count)

            # Number of updates per step in environment
            for i in range(args.updates_per_step): # environment에 1step 인가하기 전에 여러번 update를 수행
                # Update parameters of all the networks
                agent.update_parameters(memory, max_idx, args.batch_size, updates)
                updates += 1

        # add action
        act_buf.append(torch.FloatTensor(action))
        I_next  = torch.concatenate([next_state.unsqueeze(0), torch.FloatTensor(act_buf[-d_sample:]).unsqueeze(0)], dim = 1)

        if terminated or truncated:
            done = True

        episode_steps  += 1
        total_numsteps += 1
        episode_reward += reward

        I      = I.squeeze()
        I_next = I_next.squeeze()

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory.push(I, action, reward, I_next, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:

        avg_reward = 0.0
        episodes   = 10
        for _  in range(episodes):
            state = env.reset()[0]
            state = torch.FloatTensor(state)

            eval_buf = []
            for _ in range(d_sample):
                eval_buf.append(torch.FloatTensor([0]))

            episode_reward = 0
            done = False
            while not done:

                I = torch.concatenate([state.unsqueeze(0), torch.FloatTensor(eval_buf[-d_sample:]).unsqueeze(0)], dim=1)

                # TODO : Selecting action, according to voting ...
                action = agent.select_action(I, total_numsteps, evaluate=True)[0]

                next_state, reward, terminated, truncated, _ = env.step(eval_buf[-d_sample].numpy())  # Step
                next_state = torch.FloatTensor(next_state)
                eval_buf.append(torch.FloatTensor(action))

                if terminated or truncated:
                    done = True

                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()