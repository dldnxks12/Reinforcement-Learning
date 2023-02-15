import sys
import gym
import random
import numpy as np

# TD Prediction for Q function Q(s, a) <- Q(s, a) + a * (r + gamma*Q(s_, a_) - Q(s, a))

env = gym.make("FrozenLake-v1")
env = gym.wrappers.TimeLimit(env, max_episode_steps= 20)

def td_prediction(env, n_episode, random_policy):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    alpha = 0.1
    gamma = 0.9

    for _ in range(n_episode):
        state = env.reset()[0]
        terminated, truncated = False, False

        while (truncated == False) or (terminated == False):
            action = np.random.choice(np.arange(len(random_policy[state])), p = random_policy[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = np.random.choice(np.arange(len(random_policy[next_state])), p = random_policy[next_state])

            if terminated or truncated:
                Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
            else:
                Q[state][action] = Q[state][action] + alpha * ( reward + gamma*Q[next_state][next_action] - Q[state][action] )

            state = next_state

    return Q

random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
Q = td_prediction(env, 5000, random_policy)
print(Q)