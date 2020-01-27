import gym
import transfer_rl.my_env
from transfer_rl.models import PPO, Buffer
from transfer_rl.utils import parse_arg
import torch
import sys
import pickle
import os
import numpy as np

device = "cpu"
results = []
os.makedirs('results', exist_ok=True)           # Ensure these directories exist to save into
os.makedirs('trained_models', exist_ok=True)

if __name__ == "__main__":
    params = parse_arg()
    print(params)

    if params['model'] == 'ppo':
        model = PPO()
        mem = Buffer()
    else:
        sys.exit("Model type {} not recognized".format(params['model']))

    env = gym.make(params['env'], leg_length=params['leg_length'])
    model.create_model(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                       hidden_layers=params['hidden_layers'], action_std=params['action_std'],
                       learning_rate=params['learning_rate'], gamma=params['gamma'],
                       train_steps=params['train_steps'], optimizer_type=params['optimizer'],
                       device=device)

    if params['initial_model'] != 'none':
        model.load_model('./trained_models/{}.pt'.format(params['initial_model']))

    total_steps = 0
    batch_reward = 0
    batch_steps = 0
    cur_episode = 0
    while total_steps < params['total_frames']:
        cur_episode += 1
        state = env.reset()
        cur_reward = 0
        cur_steps = 0
        while cur_steps < params['max_time_steps']:
            cur_steps += 1
            if params['render']:
                env.render()

            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action, logp = model.sample_action(state)
            prev_state = state
            state, reward, done, _ = env.step(action.numpy().squeeze())
            mem.push(action, prev_state, logp, reward, done)
            cur_reward += reward

            # Train once memory is larger enough. Don't train if rendering.
            if (len(mem) >= params['batch_size']) and params['render']==False:
                model.train(mem)
                mem.reset()

            if done:
                break

        batch_steps += cur_steps
        total_steps += cur_steps
        batch_reward += cur_reward

        results.append([total_steps, cur_steps, cur_reward])

        if cur_episode % 20 == 0:
            print("{}, {}, {}".format(cur_episode, batch_reward // 20, batch_steps // 20))
            batch_reward = 0
            batch_steps = 0

        if cur_episode % 100 == 0:
            model.save_model("./trained_models/{}.pt".format(params['save_name']))

    pickle.dump(results, open('./results/' + params['save_name'] + '.p', 'wb'))
    model.save_model("./trained_models/{}.pt".format(params['save_name']))