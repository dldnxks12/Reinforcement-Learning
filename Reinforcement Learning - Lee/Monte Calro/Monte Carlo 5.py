import gym
import sys
import numpy as np
import random

env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=50)

# Define Epsilon Greedy
def epsilon_greedy(state, Q, epsilon):

    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state,:])

    return action

def generate_episode(Q, epsilon, env):

    states, actions, rewards = [], [], []

    state = env.reset()[0]

    while True:
        states.append(state)
        action = epsilon_greedy(state, Q, epsilon)
        state, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    return states, actions, rewards

gamma = 0.9
def first_visit_mc_prediction(env, epsilon, n_episodes):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    m = np.zeros([env.observation_space.n, env.action_space.n])

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(Q, epsilon, env)
        Pairs = list(zip(states, actions))
        G = 0
        for idx, (state, action, reward) in enumerate(zip(reversed(states), reversed(actions), reversed(rewards))):
            G = gamma*G + reward
            pair = (state, action)
            if pair not in Pairs[idx+1 : -1]:
                Q[state][action] += G
                m[state][action] += 1
            else:
                continue

    return Q

Q = first_visit_mc_prediction(env, 0.1, 5000)

print(Q)

#terminated, truncated = False, False
#state = env.reset()[0]

#while not terminated or not truncated:
#    action = np.argmax(Q[state]) # Optimal Policy
#    next_state, reward, terminated, truncated = env.step(action)
#    state = next_state

