import sys
import gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode = 'human')

# We use discount_factor = 1 in episodic mdp (in other words, finite horizontal mdp)
def compute_value_function(policy, discount_factor = 1):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    theta = 0.00001

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            q = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for transition_prob, next_state, reward, done in env.P[s][a]:
                    q[a] += transition_prob * (reward + np.max(Q[next_state]))

                delta = max(delta, np.abs(q[a] - Q[s][a]))
            for i in range(env.action_space.n):
                Q[s][i] = q[i]

        if delta < theta:
            break
    return Q


def extract_policy(value_table, discount_factor = 1):
    policy = np.argmax(value_table, axis = 1)
    return policy

def policy_iteration(env, discount_factor = 1):

    random_policy = np.ones(env.action_space.n) / env.action_space.n # 1/4
    no_of_iterations = 200000
    for i in range(no_of_iterations):
        new_value_function = compute_value_function(random_policy)
        new_policy = extract_policy(new_value_function)

        if(np.all(random_policy == new_policy)):
            break

        random_policy = new_policy

    return new_policy, new_value_function



optimal_policy, new_value_function = policy_iteration(env, discount_factor = 1)

print(new_value_function)
print(optimal_policy)

