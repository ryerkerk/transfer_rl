import gym
from transfer_rl.utils import parse_arg, setup_env, setup_model_mem
from transfer_rl import EpisodeRunner
import sys
import pickle
import os
import time
import numpy as np

device = "cpu"
os.makedirs('results', exist_ok=True)           # Ensure these directories exist to save into
os.makedirs('trained_models', exist_ok=True)

if __name__ == "__main__":
    params = parse_arg()
    print(params)

    env = setup_env(params)
    model, mem = setup_model_mem(env, params, device=device)

    if params['initial_model'] != 'none':
        print("Loading {}".format('./trained_models/{}.pt'.format(params['initial_model'])))
        model.load_model('./trained_models/{}.pt'.format(params['initial_model']))

    total_steps = 0
    runner = EpisodeRunner(env, model, mem, params['save_name'], device=device, render=params['render'])

    runner.run(params['total_frames'])