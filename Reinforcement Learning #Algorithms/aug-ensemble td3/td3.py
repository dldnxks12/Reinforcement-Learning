import utils
import torch
import random
import numpy as np
import network
import buffer


class TD3:
    def __init__(self, state_dim, action_dim, action_bound, device, d_sample, net_name, max_step, num_model = 5):

        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.action_bound = action_bound
        self.d_sample     = d_sample    # number of constant delayed sample
        self.net_name     = net_name
        self.num_model    = num_model

        self.device       = device
        self.capacity     = 1000000
        self.memory       = buffer.ReplayMemory(self.state_dim, self.action_dim, device, self.capacity)
        self.batch_size   = 128

        self.act_noise_scale = 0.1
        self.actor_lr        = 0.001
        self.critic_lr       = 0.001
        self.gamma           = 0.99
        self.tau             = 0.005
        self.max_step        = max_step

        if self.net_name == 'simple':
            # Js - simple network
            self.actor         = network.Actor(self.state_dim, self.action_dim, self.action_bound, self.device).to(self.device)
            self.critic        = network.Critic(self.state_dim, self.action_dim, self.device).to(self.device)
            self.target_actor  = network.Actor(self.state_dim, self.action_dim, self.action_bound, self.device).to(self.device)
            self.target_critic = network.Critic(self.state_dim, self.action_dim, self.device).to(self.device)

        elif self.net_name == 'ensemble':
            self.actor_list         = []
            self.target_actor_list  = []

            h_dim =  128
            for i in range(self.num_model):
                actor         = network.Actor(self.state_dim, self.action_dim, self.action_bound, self.device, hidden_dim =h_dim * (i + 1)).to(self.device)
                target_actor  = network.Actor(self.state_dim, self.action_dim, self.action_bound, self.device, hidden_dim =h_dim * (i + 1)).to(self.device)

                self.actor_list.append(actor)
                self.target_actor_list.append(target_actor)

            self.critic        = network.Critic(self.state_dim, self.action_dim, self.device).to(self.device)
            self.target_critic = network.Critic(self.state_dim, self.action_dim, self.device).to(self.device)

        elif self.net_name == 'noisy':
            # Js - noisy network
            self.actor         = network.NOISYActor(self.state_dim, self.action_dim, self.action_bound, self.device).to(self.device)
            self.critic        = network.NOISYCritic(self.state_dim, self.action_dim, self.device).to(self.device)
            self.target_actor  = network.NOISYActor(self.state_dim, self.action_dim, self.action_bound, self.device).to(self.device)
            self.target_critic = network.NOISYCritic(self.state_dim, self.action_dim, self.device).to(self.device)

        elif self.net_name == 'jang':
            # Jangs network
            self.actor         = network.Policy(self.state_dim, self.action_dim, self.action_bound, self.device)
            self.critic        = network.Twin_Q_net(self.state_dim, self.action_dim, self.device)
            self.target_actor  = network.Policy(self.state_dim, self.action_dim, self.action_bound, self.device)
            self.target_critic = network.Twin_Q_net(self.state_dim, self.action_dim, self.device)

        else:
            raise Exception("Invalid network")

        # Optimizer Setting
        if self.net_name == 'ensemble':
            self.actor_optimizer_list = []

            for i in range(self.num_model):
                self.actor_optimizer_list.append(torch.optim.Adam(self.actor_list[i].parameters(), lr=self.actor_lr))
                utils.hard_update(self.actor_list[i], self.target_actor_list[i])

            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
            utils.hard_update(self.critic, self.target_critic)

        else:
            self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

            utils.hard_update(self.actor, self.target_actor)
            utils.hard_update(self.critic, self.target_critic)


    def get_action(self, state, total_step, add_noise = True): # adding noise decaying
        with torch.no_grad():
            if self.net_name == 'ensemble':
                evals       = []
                action_list = []
                action_idx  = [i for i in range(self.num_model)]

                if add_noise: # Train
                    for i in range(self.num_model):
                        noise = np.random.normal(loc=0, scale=abs(self.action_bound[1]) * self.act_noise_scale, size=self.action_dim)
                        action = self.actor_list[i](state).cpu().numpy()[0] + noise
                        action = np.clip(action, self.action_bound[0], self.action_bound[1])
                        action_list.append(action)

                        eval_  = self.critic.Q_A(state.to(self.device), action)
                        evals.append(eval_)

                    evaluations    = torch.stack(evals)

                    # max_step이 됬을 때 epsilon == 0
                    if random.random() < ((self.max_step - total_step) / self.max_step) : # softmax voting
                        action_softmax = torch.nn.functional.softmax(evaluations, dim = 0).squeeze()
                        choice_action  = np.random.choice(action_idx, 1, p=action_softmax.cpu().detach().numpy())
                        action = action_list[choice_action[0]]

                    else: # argmax
                        choice_action = torch.argmax(evaluations)
                        action = action_list[choice_action.item()]

                else: # eval
                    for i in range(self.num_model):
                        action = self.actor_list[i](state).cpu().numpy()[0]
                        action_list.append(action)

                        eval_  = self.critic.Q_A(state.to(self.device), action)
                        evals.append(eval_)

                    evaluations    = torch.stack(evals)
                    choice_action = torch.argmax(evaluations)
                    action = action_list[choice_action.item()]

                return action, choice_action.item()

            else:
                if add_noise:
                    noise  = np.random.normal(loc=0, scale=abs(self.action_bound[1]) * self.act_noise_scale, size=self.action_dim)
                    action = self.actor(state).cpu().numpy()[0] + noise
                    action = np.clip(action, self.action_bound[0], self.action_bound[1])
                else:
                    action = self.actor(state).cpu().numpy()[0]

                return action

    def train_actor(self, states): # OK
        if self.net_name == 'ensemble':
            for i in range(self.num_model):
                actor_loss = 0
                actor_loss = -self.critic.Q_A(states, self.actor_list[i](states)).mean()
                self.actor_optimizer_list[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizer_list[i].step()

        else:
            actor_loss = -self.critic.Q_A(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

    def train_critic(self, states, actions, rewards, next_states, dones, idx):

        if self.net_name == 'ensemble':
            target_act_noise = (torch.randn_like(actions) * 0.2).clamp(-0.2, 0.2).to(self.device)

            with torch.no_grad():
                next_target_action = (self.target_actor_list[idx](next_states) + target_act_noise).clamp(self.action_bound[0], self.action_bound[1])
                NQA, NQB           = self.target_critic(next_states, next_target_action)
                next_q_values      = torch.min(NQA, NQB)
                target_q_values    = rewards + (1 - dones) * self.gamma * next_q_values

            QA, QB      = self.critic(states, actions)
            critic_loss = ((QA - target_q_values) ** 2).mean() + ((QB - target_q_values) ** 2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        else:
            with torch.no_grad():
                target_act_noise   = (torch.randn_like(actions) * 0.2).clamp(-0.2, 0.2).to(self.device)
                next_target_action = (self.target_actor(next_states) + target_act_noise).clamp(self.action_bound[0], self.action_bound[1])

                NQA, NQB = self.target_critic(next_states, next_target_action)
                next_q_values   = torch.min(NQA, NQB)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            QA, QB = self.critic(states, actions)
            critic_loss = ((QA - target_q_values)**2).mean() + ((QB - target_q_values)**2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def train(self, idx, option = 'both'):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        if option == 'both':
            self.train_actor(states)
            self.train_critic(states, actions, rewards, next_states, dones, idx)

            if self.net_name == 'ensemble':
                for i in range(self.num_model):
                    utils.soft_update(self.actor_list[i], self.target_actor_list[i], self.tau)
                utils.soft_update(self.critic, self.target_critic, self.tau)

            else:
                utils.soft_update(self.actor, self.target_actor, self.tau)
                utils.soft_update(self.critic, self.target_critic, self.tau)

        elif option == 'critic_only':
            self.train_critic(states, actions, rewards, next_states, dones, idx)

        else:
            raise Exception("Invalid option")
