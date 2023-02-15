import gym
import sys
import random
import numpy as np

#env = gym.make('Taxi-v3')
env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps = 50)

"""
# Solve Taxi problem with Double Q-Learning (Offline TD(0) with Soft greedy policy)

    -> Remove maximization bias  

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


def soft_greedy_policy(Q1, Q2, state):
    e = 0.3
    if random.random() < e:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q1[state,:] + Q2[state,:]) # return index

    return action

def td_control(env, n_episode):
    # Double Q Learning을 위해 2개의 Q function을 정의
    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = np.zeros((env.observation_space.n, env.action_space.n))

    alpha = 0.1
    gamma = 0.9

    random_list = [0, 1]

    for _ in range(n_episode):
        state = env.reset()[0]
        terminated, truncated = False, False

        while (terminated == False) or (truncated == False):
            action = soft_greedy_policy(Q1, Q2, state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            choiced_network = random.choice(random_list)

            if choiced_network == 0:
                if terminated or truncated:
                    Q1[state][action] += alpha * (reward - Q1[state][action])
                else:
                    max_action = np.argmax(Q1[next_state,:])
                    Q1[state][action] += alpha * (reward + (gamma * Q2[next_state, max_action]) - Q1[state][action])
            else:
                if terminated or truncated:
                    Q2[state][action] += alpha * (reward - Q2[state][action])
                else:
                    max_action = np.argmax(Q2[next_state,:])
                    Q2[state][action] += alpha * (reward + (gamma * Q1[next_state, max_action]) - Q2[state][action])

            state = next_state
    return Q1, Q2

Q1, Q2 = td_control(env, 5000)