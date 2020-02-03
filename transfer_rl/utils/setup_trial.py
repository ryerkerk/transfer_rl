import gym
from ..models import PPO, DDPG, Buffer
import sys

def setup_env(params):
    if params['env'] == 'Bipedal_Custom_Leg_Length-v0':
        env = gym.make('Bipedal_Custom_Leg_Length-v0',
           max_steps=params['max_time_steps'],
           leg_length=params['leg_length'],
           terrain_length_scale=params['terrain_length_scale'],
           fall_penalty=params['fall_penalty'],
           finish_bonus=params['finish_bonus'],
           torque_penalty=params['torque_penalty'],
           head_balance_penalty=params['head_balance_penalty'],
           head_height_penalty=params['head_height_penalty'],
           leg_sep_penalty=params['leg_sep_penalty'],
           torque_diff_penalty=params['torque_diff_penalty'])
    else:
        sys.exit("Environment type {} not recognized".format(params['env']))

    return env


def setup_model_mem(env, params, device='mem'):
    if params['model'] == 'ppo':
        model = PPO(device=device)
        model.create_model(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                           hidden_layers=params['hidden_layers'], action_std=params['action_std'],
                           learning_rate=params['learning_rate'], gamma=params['gamma'],
                           batch_size=params['batch_size'], train_steps=params['train_steps'],
                           optimizer_type=params['optimizer'])
        mem = Buffer()
    elif params['model'] == 'ddpg':
        model = DDPG(device=device)
        model.create_model(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                           hidden_layers=params['hidden_layers'], action_std=params['action_std'],
                           action_std_decay=params['action_std_decay'],
                           learning_rate=params['learning_rate'], gamma=params['gamma'],
                           batch_size=params['batch_size'], train_steps=params['train_steps'],
                           optimizer_type=params['optimizer'])
        mem = Buffer(capacity = params['memory_capacity'])
    else:
        sys.exit("Model type {} not recognized".format(params['model']))

    return model, mem
