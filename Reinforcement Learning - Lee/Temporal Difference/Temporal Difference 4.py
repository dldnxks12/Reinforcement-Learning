import gym
import sys
import random
import numpy as np

#env = gym.make('Taxi-v3')
env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps = 50)

"""
# Solve Taxi problem with Q-Learning (Offline TD(0) with Soft greedy policy) 

state : 
 - taxi_row 
 - taxi_col
 - passenger_location 
 - destination

action : 
 - 0: move south
 - 1: move north
 - 2: move east 
 - 3: move west 
 - 4: pickup passenger
 - 5: dropoff passenger
"""


def soft_greedy_policy(Q, state):
    e = 0.3
    if random.random() < e:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :]) # return index
    return action

def td_control(env, n_episode):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    alpha = 0.1
    gamma = 0.9

    for _ in range(n_episode):
        state = env.reset()[0]
        terminated, truncated = False, False

        while (terminated == False) or (truncated == False):
            action = soft_greedy_policy(Q, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            #next_action = soft_greedy_policy(Q, next_state) -> Q-Learning에서는 Next action을 max operation으로 처리

            if terminated or truncated:
                Q[state][action] += alpha * (reward - Q[state][action])
            else:
                Q[state][action] += alpha * (reward + (gamma * max(Q[next_state,: ]) - Q[state][action]))

            state = next_state
    return Q

Q = td_control(env, 5000)
print(Q)