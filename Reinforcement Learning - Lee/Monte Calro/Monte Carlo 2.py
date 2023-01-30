"""

 Every Visit Monte Carlo Prediction
 1. Batch version
 2. Recursive version - incremental mean

"""

import sys
import gym
import numpy as np

# Define Environment
env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=50)

# Generate Episode
def generate_episode(policy, env):
    states, actions, rewards = [], [], []

    observation = env.reset()[0]
    while True:
        states.append(observation)
        probs = policy[observation]
        action = np.random.choice(np.arange(len(probs)), p = probs)
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    return states, actions, rewards

gamma = 0.9

# Batch Version
def first_visit_monte_carlo_prediction_Batch(env, random_policy, n_episodes):
    V = np.zeros(env.observation_space.n)
    m = np.zeros(env.observation_space.n)

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(random_policy, env)
        G = 0
        for idx, (state, reward) in enumerate(zip(reversed(states), reversed(rewards))):
            G = (G*gamma) + reward
            if state not in states[idx+1:-1]:
                m[state] += 1
                V[state] += G
            else:
                continue

    for i in range(len(states)):
        if m[i] == 0:
            continue
        V[i] = V[i] / m[i]
    return V


# Recursive Version
def first_visit_monte_carlo_prediction_Recursive(env, random_policy, n_episodes):
    V = np.zeros(env.observation_space.n)
    m = np.zeros(env.observation_space.n)

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(random_policy, env)
        G = 0
        for idx, (state, reward) in enumerate(zip(reversed(states), reversed(rewards))): # For every state
            G = (G*gamma) + reward
            if state not in states[idx+1:-1]:
                m[state] += 1
                V[state] = V[state] + (1/m[state])*(G - V[state])
            else:
                continue

    return V

random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
V_r = first_visit_monte_carlo_prediction_Recursive(env, random_policy, n_episodes = 5000)
V_b = first_visit_monte_carlo_prediction_Batch(env, random_policy, n_episodes = 5000)

print(V_r)
print(V_b)


