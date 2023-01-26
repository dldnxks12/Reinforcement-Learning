# 코드 안돌아감 컨셉만 이해하기

import torch

def Epsilon_greedy_policy(N_actions, Epsilon):
    def policy_function(state, Q): # state와 Q value
        probs               = (torch.ones(N_actions) * Epsilon) / N_actions # e/m
        best_action         = torch.argmax(Q[state]).item() # best action의 index return
        probs[best_action] += 1-Epsilon
        action              = torch.multinomial(probs, 1).item() # Random selection according to probs
        return action

    return policy_function

def SARSA(env, gamma, n_episode, alpha):

    n_states = 64
    n_actions = env.action_space.n # number of actions
    Q = torch.FloatTensor([[0,0,0,0]*64])

    for episode in range(n_episode):
        total_reward = 0
        state   = env.reset()
        is_done = False  # Terminal state
        action  = Epsilon_greedy_policy(state, Q)

        while not is_done:
            next_state, reward, is_done, info = env.step(action)
            next_action = Epsilon_greedy_policy(next_state, Q)

            # TD Error
            td_delta = reward + gamma*Q[next_state][next_action] - Q[state][action]

            # Q Update
            Q[state][action] = Q[state][action] + (alpha * td_delta)
            total_reward += reward

            if is_done:
                break
            state = next_state
            action = next_action

        policy = {}
        for state, action in Q.items():
            policy[state] = torch.argmax(action).item()

        return Q, policy


