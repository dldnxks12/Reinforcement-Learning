import sys
import gym
import random
import numpy as np


# TD Prediction for value function V <- V + a*(r + gamma*V_next - V)
# Policy -> uniformly random policy

env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=50)

def td_prediction(env, n_epsiode, random_policy):
    V = np.zeros(env.observation_space.n)
    alpha = 0.1
    gamma = 0.9

    for _ in range(n_epsiode):

        state = env.reset()[0]
        terminated, truncated = False, False

        while (terminated == False) or (truncated == False):
            action = np.random.choice(np.arange(len(random_policy[state])), p = random_policy[state])
            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                V[state] = V[state] + alpha*( reward - V[state])
            else:
                V[state] = V[state] + alpha * ((reward + gamma * V[next_state]) - V[state])
            state = next_state

    return V

random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
V = td_prediction(env, 10000, random_policy)
print(V)








