import gymnasium as gym
import torch
import utils
import os
import td3
import trainer
import environment

def main():

    device        = 'cuda' if torch.cuda.is_available() else 'cpu'
    device        = 'cpu'
    random_seed   = 1

    seed = utils.set_seed(random_seed)

    # env, env_eval = utils.make_env(env_name, seed)
    #env_name = "Hopper-v4"

    xml_file = os.getcwd() + "/environment/assets/inverted_single_pendulum.xml"

    env_name = "InvertedSinglePendulum-v4"
    env = gym.make("InvertedSinglePendulum-v4", model_path=xml_file)
    env_eval = gym.make("InvertedSinglePendulum-v4", model_path=xml_file) #, render_mode = 'human')
    print(f"Device : {device} Random Seed : {seed} Environment : {env_name}")

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low[0], env.action_space.high[0]]

    agent = td3.TD3(state_dim, action_dim, action_bound, device)

    train = trainer.Trainer(env, env_eval, agent, env_name, device)
    train.run()

    return env, env_eval, agent, env_name, device

if __name__ == "__main__":
    env, env_eval, agent, env_name, device = main()
