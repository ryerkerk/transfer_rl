import gym
from ..models import PPO, Buffer
import sys


def setup_env(params):
    """
    Sets up the environment in which the model will be trained. At the moment only
    a bipedal walker is considered.

    Environments need to fit the OpenAi gym API.
    """

    if params['env'] == 'Bipedal_Custom_Leg_Length-v0':
        env = gym.make('Bipedal_Custom_Leg_Length-v0',
           max_steps=params['max_time_steps'],
           leg_length=params['leg_length'],
           terrain_length_scale=params['terrain_length_scale'],
           fall_penalty=params['fall_penalty'] )
    else:
        sys.exit("Environment type {} not recognized".format(params['env']))

    return env


def setup_model_mem(env, params, device='mem'):
    """
    Initialize the reinforcement learning model and memory buffer. If transfer learning is
    applied then a transfer learning model will be setup within the RL model.

    Adaptive action noise will also be initialized if the appropriate parameter values are
    set (action_std_start, action_std_final, and action_std_end).

    Currently only a ppo model is used.
    """

    # Initialize model and memory.
    if params['model'] == 'ppo':
        model = PPO(device=device)
        model.create_model(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                           hidden_layers=params['hidden_layers'], action_std=params['action_std'],
                           learning_rate=params['learning_rate'], gamma=params['gamma'],
                           batch_size=params['batch_size'], train_steps=params['train_steps'],
                           optimizer_type=params['optimizer'], total_frames=params['total_frames'],
                           transfer_learning=params['transfer_learning'], tl_end=params['tl_end'])
        mem = Buffer()  # Initialize memory buffer.
    else:
        sys.exit("Model type {} not recognized".format(params['model']))

    # If adaptive action noise is used then call appropriate model function.
    if params['action_std_start'] >= 0:
        print("Using adaptive action noise")
        model.set_adaptive_action_std(action_std_start=params['action_std_start'],
                                      action_std_final=params['action_std_final'],
                                      action_std_end=params['action_std_end'],
                                      total_frames=params['total_frames'])

    return model, mem

