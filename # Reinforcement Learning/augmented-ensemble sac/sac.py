import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from network import GaussianPolicy, QNetwork

class SAC(object):
    def __init__(self, num_inputs, action_space, delay, num_model, args):
        self.alpha     = args.alpha  # temperature
        self.gamma     = args.gamma  # discount factor for infinite horizon case
        self.tau       = args.tau    # update
        self.num_model = num_model
        self.max_step  = args.num_steps

        self.target_update_interval   = args.target_update_interval
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("----------------------------------------------------------------------------------------")
        print(f"# - On {self.device} | temperature : {self.alpha} | d_sample : {delay[0]} | sampling_time : {delay[1]} | num_model : {self.num_model}")
        print("----------------------------------------------------------------------------------------")

        # TODO 1 : Define Ensemble Networks and optimizer, here
        # TODO 2 : Check

        # Actor x num_models
        self.policy_list       = []
        self.policy_optim_list = []

        actor_hidden_size = args.hidden_size // 2

        for i in range(self.num_model):
            policy        = GaussianPolicy(num_inputs, action_space.shape[0], actor_hidden_size * (i+1), action_space).to(self.device)
            policy_optim  = Adam(policy.parameters(), lr=args.lr)
            self.policy_list.append(policy)
            self.policy_optim_list.append(policy_optim)

        # Critic x 1
        self.critic        = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.lr)

        hard_update(self.critic_target, self.critic)  # parameter 동기화

    def select_action(self, state, total_numsteps, evaluate=False):
        evals = []
        action_list = []
        action_idx = [i for i in range(self.num_model)]

        state = torch.FloatTensor(state).to(self.device) #.unsqueeze(0)

        for i in range(self.num_model):
            if evaluate is False:
                action, _, _ = self.policy_list[i].sample(state) #
            else:
                _, _, action = self.policy_list[i].sample(state) # action => mean

            action = action.detach().cpu().numpy()
            action_list.append(action)

            # TODO : compare - get eval_ with using 1. self.critic 2. self.critic_target
            eval_, _ = self.critic(state, torch.FloatTensor(action).to(self.device))
            # eval_ = self.critic_target(state, action)
            evals.append(eval_)

        evaluations = torch.stack(evals)

        # softmax
        if random.random() < ((self.max_step - total_numsteps) / self.max_step):
            action_softmax = torch.nn.functional.softmax(evaluations, dim = 0).squeeze()
            choice_action  = np.random.choice(action_idx, 1, p = action_softmax.cpu().detach().numpy())
            action         = action_list[choice_action[0]]
        # argmax
        else:
            choice_action = torch.argmax(evaluations)
            action        = action_list[choice_action.item()]

        return action[0], choice_action.item()


    # TODO : update target actor / critic with consideration with ensemble networks
    def update_parameters(self, memory, max_idx, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch      = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch     = torch.FloatTensor(action_batch).to(self.device)
        reward_batch     = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch       = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)


        # TODO : Critic update with respect to max_idx actor (policy_list[max_idx])
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy_list[max_idx].sample(next_state_batch)
            qf1_next_target, qf2_next_target        = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value       = reward_batch + self.gamma * (min_qf_next_target) * mask_batch

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # TODO : Actor update with respect to all actors (policy_list[:])
        for i in range(self.num_model):
            policy_loss = 0
            pi, log_pi, _ = self.policy_list[i].sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim_list[i].zero_grad()
            policy_loss.backward()
            self.policy_optim_list[i].step()

        # Soft update of Critic
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict'          : self.policy.state_dict(),
                    'critic_state_dict'          : self.critic.state_dict(),
                    'critic_target_state_dict'   : self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(        checkpoint['policy_state_dict'])
            self.critic.load_state_dict(        checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict( checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(  checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(  checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()