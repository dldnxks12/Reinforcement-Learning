import sys
import utils
import torch
import numpy as np
import random

class Trainer:
    def __init__(self, env, env_eval, agent, env_name, device, d_sample, max_step, num_model, ensemble = False):

        self.device   = device
        self.agent    = agent
        self.env_name = env_name
        self.env      = env
        self.env_eval = env_eval
        self.d_sample = d_sample    # number of constant delayed sample
        self.ensemble = ensemble
        self.num_model= num_model

        self.start_step   = 10000
        self.update_after = 1000
        self.max_step     = max_step
        self.batch_size   = 128
        self.update_every = 50
        self.policy_update_delay = 2

        self.eval_flag          = True
        self.eval_episode       = 5
        self.eval_freq          = 5000

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.finish_flag = False

        self.recorder_reward   = []
        self.recorder_timestep = []


    def evaluate(self):
        # Evaluate process
        reward_list = []
        for epi in range(self.eval_episode):
            epi_reward = 0
            state      = self.env_eval.reset()[0]
            state      = torch.FloatTensor(state)
            done       = False

            act_buf = []
            d = self.d_sample
            for _ in range(self.d_sample):
                act_buf.append(torch.FloatTensor([0]))

            while not done:
                I = torch.concatenate([state.unsqueeze(0), torch.FloatTensor(act_buf[-d:]).unsqueeze(0)], dim=1)

                action, _ = self.agent.get_action(I, self.total_step, add_noise=False)

                next_state, reward, terminated, truncated, _ = self.env_eval.step(act_buf[-d].numpy())
                next_state  = torch.FloatTensor(next_state)

                act_buf.append(torch.FloatTensor(action))

                done        = terminated or truncated
                state       = next_state

                if reward <= 0 :
                    reward = 0
                else:
                    reward = (reward / 10)

                epi_reward += reward

            reward_list.append(epi_reward)
        self.recorder_reward.append(sum(reward_list)/len(reward_list))
        self.recorder_timestep.append(self.total_step)

        print("total_step {}  |  episode {}  |  Avg Reward {:.2f}  |  Max reward: {:.2f}  |  "
              "Min reward: {:.2f}".format(self.total_step, self.episode, sum(reward_list)/len(reward_list),
                                               max(reward_list), min(reward_list)))

    def run(self):
        while not self.finish_flag:
            self.episode       += 1
            self.episode_reward = 0
            self.local_step     = 0

            state = self.env.reset()[0]
            state = torch.FloatTensor(state)

            done = False

            act_buf = []
            d = self.d_sample
            for _ in range(self.d_sample):
                act_buf.append(torch.FloatTensor([0]))

            idx_box = []
            while not done:
                self.local_step += 1
                self.total_step += 1

                I = torch.concatenate([state.unsqueeze(0), torch.FloatTensor(act_buf[-d:]).unsqueeze(0)], dim=1)

                if self.total_step >= self.start_step:
                    action, action_idx = self.agent.get_action(I, self.total_step, add_noise=True)
                    action = np.array(action)
                    next_state, reward, terminated, truncated, _ = self.env.step(act_buf[-d].numpy())
                    next_state = torch.FloatTensor(next_state)
                    idx_box.append(action_idx)

                else:
                    action = self.env.action_space.sample()
                    next_state, reward, terminated, truncated, _ = self.env.step(act_buf[-d].numpy())
                    next_state = torch.FloatTensor(next_state)
                    action_idx = random.randint(0, self.num_model-1)
                    idx_box.append(action_idx)

                act_buf.append(torch.FloatTensor(action))
                I_next  = torch.concatenate([next_state.unsqueeze(0), torch.FloatTensor(act_buf[-d:]).unsqueeze(0)], dim = 1)

                done = terminated or truncated
                done_mask = 0.0 if self.local_step == self.env._max_episode_steps else float(done)

                self.agent.memory.push(I, action, reward, I_next, done_mask)

                state = next_state

                self.episode_reward += reward

                # update
                if self.agent.memory.size >= self.batch_size and self.total_step >= self.update_after and self.total_step % self.update_every == 0:
                    idx = max(idx_box, key = idx_box.count)
                    for i in range(self.update_every):
                        if (i+1) % self.policy_update_delay:
                            self.agent.train(idx, option='both')  # train actor and critic both.
                        else:
                            self.agent.train(idx, option='critic_only')  # train critic only

                # Evaluation
                if self.eval_flag and self.total_step % self.eval_freq == 0:
                    self.evaluate()

                # Raise finish_flag.
                if self.total_step == self.max_step:

                    reco_reward   = np.array(self.recorder_reward)
                    reco_timestep = np.array(self.recorder_timestep)

                    np.save('../recorder/td3_ensemble_model3_6', reco_reward)

                    self.finish_flag = True

