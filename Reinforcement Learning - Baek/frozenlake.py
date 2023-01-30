import numpy as np
import gym
import sys

env = gym.make('FrozenLake-v1')

print(env)

"""
def policy_eval(policy, env, discount_factor = 1.0, theta = 0.00001):
    V = np.zeros(env.nS) # state Space 개수 만큼 0 행렬 생성

    while True:
        delta = 0
        # for each state, perform a 'Full Back up'
        for s in range(env.nS):
            v = 0

            Total_Temp = 0
            for a, action_probability in enumerate(policy[s]): # 해당 s에서 취할 수 있는 4개의 action ..
                # env.P[s][a]은 s에서 Action a를 취했을 때 나타날 다음 State의 확률을 의미 ..

                Sum_Temp = 0
                Reward_Temp = 0
                for transition_probability, next_state, reward, done in env.P[s][a]:

                    # reward : immediate reward = r(s, a, s')
                    # transition_probability = P(s' | s,a)
                    # next_state = s'

                    Sum_Temp += transition_probability * V[next_state] # 갈 수 있는 모든 State에 대해서 Value 모아두기
                    Reward_Temp += reward

                Reward_Temp /= 4 # reward function
                Total_Temp += action_probability * (Reward_Temp + Sum_Temp)

            v = Total_Temp

            # 계산한 v와 이전에 저장해둔 V[s]와 차이가 일정량 이하면 업데이트 종료!
            delta = max(delta, np.abs(v - V[s])) # Check
            V[s] = v

        if delta < theta:
            break

    return np.array(V)

# Policy Improvement를 할 시 이 부분을 Greedy Policy로 설정...
random_policy = np.ones([env.nS, env.nA]) / env.nA # 1/4의 확률로 Setting --- 16 x 4
v = policy_eval(random_policy, env)

print(f"Value Function : {v}")
print("")
"""

