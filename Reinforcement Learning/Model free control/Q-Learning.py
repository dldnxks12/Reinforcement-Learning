import torch
import gym

env = gym.make("FrozenLake8x8-v1", is_slippery=True)

def Epsilon_greedy_policy(N_actions, Epsilon):
    def policy_function(state, Q): # state와 Q value
        probs               = (torch.ones(N_actions) * Epsilon) / N_actions # e/m
        best_action         = torch.argmax(Q[state]).item() # best action의 index return
        probs[best_action] += 1-Epsilon
        action              = torch.multinomial(probs, 1).item() # Random selection according to probs
        return action

    return policy_function

# Off policy
def q_learning(env, gamma, n_episode, alpha):
    n_actions = env.action_space.n
    Q = torch.zeros_liks((env.state_space.n, env.action_space.n))


    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        while not is_done:

            action = Epsilon_greedy_policy(state, Q)
            next_state, reward, is_done, info = env.step(action)
            Q_ = torch.max(Q[next_state]) # max !

            td_delta = reward + gamma*(Q_) - Q[state][action]
            Q[state][action] = Q[state][action] + alpha*td_delta

            if is_done:
                break
            state = next_state




