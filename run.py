import gym
from transfer_rl.utils import parse_arg, setup_env, setup_model_mem
from transfer_rl import EpisodeRunner
import sys
import pickle
import os
import time
import numpy as np

os.makedirs('results', exist_ok=True)  # Ensure these directories exist to save into
os.makedirs('trained_models', exist_ok=True)

if __name__ == "__main__":
    params = parse_arg()
    print(params)

    device = params['device']

    env = setup_env(params)
    model, mem = setup_model_mem(env, params, device=device)

    if params['initial_model'] != 'none':
        print("Loading {}".format('./trained_models/{}.pt'.format(params['initial_model'])))
        model.load_model('./trained_models/{}.pt'.format(params['initial_model']), params['models_to_load'])
        if params['reset_final_layer'] is not None:
            model.reset_final_layer(params['reset_final_layer'])

        if params['add_noise_layers'] >= 1:
            model.add_noise_layers(n=params['add_noise_layers'], alpha=params['add_noise_alpha'],
                                   model_type=params['models_to_add_noise'])


    total_steps = 0
    runner = EpisodeRunner(env, model, mem, params['save_name'], device=device, render=params['render'])

    runner.run(params['total_frames'])
