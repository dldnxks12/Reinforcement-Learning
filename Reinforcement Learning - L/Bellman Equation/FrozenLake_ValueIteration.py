# Solve frozen lake with Policy Iteration

import gym
import sys
import numpy as np
from time import sleep

env = gym.make('FrozenLake-v1', render_mode = "human")

def QVI(env, discount_factor = 1.0, theta = 0.00001):
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            q = np.zeros(env.action_space.n)

            for a in range(env.action_space.n):
                for transition_prob, next_state, reward, done in env.P[s][a]: # next_state로..
                    q[a] += transition_prob * (reward + np.max(Q[next_state]))
                delta = max(delta, np.abs(q[a] - Q[s][a]))


            for i in range(env.action_space.n):
                Q[s][i] = q[i]

        if delta < theta:
            break

    return np.array(Q)

q = QVI(env)

print("Q Value function :")
print(q)

# Extract Optimal Policy
Optimal_policy = np.argmax(q, axis = 1)
print("Optimal Policy : ", Optimal_policy)





