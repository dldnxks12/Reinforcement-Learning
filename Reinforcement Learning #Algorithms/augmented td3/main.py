import torch
import utils

import td3
import trainer


def main():

    device        = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed   = 10

    env_name      = "InvertedSinglePendulum-v4"
    seed          = utils.set_seed(random_seed)
    env, env_eval = utils.make_env(env_name, seed)
    d_sample      = 6
    s_time        = "50ms" # modify this at ./environment/inverted_single_pendulum_v4.py -> frame_skip , render_fps

    # network type : simple | noisy | jang
    net_name = "simple"

    print(f"Device : {device} \nDelayed Sample : {d_sample} | Sampling Time : {s_time}  | Environment : {env_name} | Network : {net_name}" )
    print("-------------------------------------------------------------------------------------------------------------------------------")
    state_dim    = env.observation_space.shape[0] + d_sample
    action_dim   = env.action_space.shape[0]
    action_bound = [env.action_space.low[0], env.action_space.high[0]]

    agent = td3.TD3(state_dim, action_dim, action_bound, device, d_sample, net_name)

    train = trainer.Trainer(env, env_eval, agent, env_name, device, d_sample)
    train.run()

if __name__ == "__main__":
    main()
