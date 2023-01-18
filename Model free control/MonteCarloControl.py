import numpy as np

# - Define 5x5 Grid World - #
class Env:
    def __init__(self):
        self.grid_width  = 5
        self.grid_height = 5
        self.action_grid = [(-1,0), (1,0), (0,-1), (0,1)] # Up Down Left Right
        self.gtriangle1  = [1, 2] # Obstacle
        self.gtriangle2  = [2, 1] # Obstacle
        self.goal        = [2, 2] # Goal Point

    def step(self, state, action):
        x, y = state # row , column

        # Get next state by action
        x += action[0]
        y += action[1]

        # Boundary Condition
        if x < 0:
            x = 0
        elif x > (self.grid_width - 1):
            x = self.grid_width-1

        if y < 0:
            y = 0
        elif y > (self.grid_height - 1):
            y = (self.grid_height - 1)

        next_state = [x, y]

        # Reward condition / done = program end flag
        if next_state == self.gtriangle1 or next_state == self.gtriangle2: # Bad Points
            reward = -1
            done = True

        elif next_state == self.goal:
            reward = 1
            done = True

        else:
            reward = 0
            done   = False

        return next_state, reward, done

    def reset(self):
        return [0, 0] # Return to starting state



# - Agent of Monte Carlo Control - #
class MC_agent:

    def __init__(self):
        self.action_grid = [(-1,0), (1,0), (0,-1), (0,1)] # Up Down Left Right
        self.action_text = ["U", "D", "L", "R"]
        self.grid_width  = 5
        self.grid_height = 5
        self.value_table = np.zeros((self.grid_width, self.grid_height))
        self.memory      = []

        # Hyperparameters
        self.e = 0.1 # Exploration variable Epsilon
        self.learning_rate = 0.01
        self.discount_factor = 0.95

    def get_action(self, state):
        # with prob.ε take random action
        if np.random.randn() < self.e:
            idx = np.random.choice(len(self.action_grid), 1)[0]
        else:
            next_values = np.array([])
            for s in self.next_states(state):
                next_values = np.append(next_values, self.value_table[tuple(s)])
            max_value = np.amax(next_values)
            tie_Qchecker = np.where(next_values == max_value)[0]

            # if tie max value, get random
            if len(tie_Qchecker) > 1:
                idx = np.random.choice(tie_Qchecker, 1)[0]
            else:
                idx = np.argmax(next_values)
        action = self.action_grid[idx]
        return action

    def next_states(self, state):
        x, y = state
        next_S = []
        for action in self.action_grid:
            # calculate x_coordinate
            x+=action[0]
            if x < 0:
                x = 0
            elif x > 4:
                x = 4
            # calculate x_coordinate
            y+=action[1]
            if x < 0:
                x = 0
            elif x > 4:
                x = 4
            next_S.append([x, y])
        return next_S

        # using First visit MC

    def update(self):
        G_t = 0
        visit_states = []
        for sample in reversed(self.memory):
            state = sample[0]
            reward = sample[1]
            if state not in visit_states:
                visit_states.append(state)
                G_t = reward + self.discount_factor * G_t
                V_t = self.value_table[tuple(state)]
                # update Value
                self.value_table[tuple(state)] = V_t + self.learning_rate * (G_t - V_t)

    def memorizer(self, state, reward, done):
        self.memory.append([state, reward, done])

    def save_actionseq(self, action_sequence, action):
        idx = self.action_grid.index(action)
        action_sequence.append(self.action_text[idx])


if __name__ == "__main__":
    env = Env()
    agent = MC_agent()
    total_episode = 10000
    sr = 0

    for episode in range(total_episode):
        action_sequence = []
        total_reward = 0
        state = env.reset()
        action = agent.get_action(state)
        done = False
        walk = 0

        while True:
            next_state, reward, done = env.step(state, action)
            agent.memorizer(state, reward, done)
            agent.save_actionseq(action_sequence, action)
            walk += 1

            # next state and action
            state = next_state
            action = agent.get_action(state)
            total_reward += reward

            if done:
                if episode % 100 == 0:
                    print('finished at', state)
                    print('episode :{}, The number of step:{}\n The sequence of action is:\
                          {}\nThe total reward is: {}\n'.format(episode, walk, action_sequence, total_reward))
                if state == env.goal:
                    sr += 1
                agent.update()
                agent.memory.clear()
                break

print('The accuracy :', sr / total_episode * 100, '%')





