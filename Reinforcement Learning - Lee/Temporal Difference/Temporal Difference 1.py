import sys
import gym
import random
import numpy as np
import torch

# TD Prediction for value function V <- V + a*(r + gamma*V_next - V)
# Policy -> uniformly random policy

env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=50)

def generate_episode(env, policy):
    states, actions, rewards, next_states = [], [], [], []
    state = env.reset()[0]

    while True:
        states.append(state)
        probs = policy[state]
        action = np.random.choice(np.arange(len(probs)), p = probs)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state
        if terminated or truncated:
            break

    return states, actions, rewards, next_states

def td_prediction(env, n_epsiode, random_policy):
    V = np.zeros(env.observation_space.n)
    alpha = 0.1
    gamma = 0.9

    for _ in range(n_epsiode):
        states, actions, rewards, next_states = generate_episode(env, random_policy)

        for state, reward, next_state in zip(states, rewards, next_states):
            V[state] = V[state] + alpha*( (reward + gamma*V[next_state]) - V[state])

    print(V)
    return V

random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
V = td_prediction(env, 10000, random_policy)








