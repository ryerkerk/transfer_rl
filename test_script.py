import gym
import transfer_rl.my_env
from transfer_rl.models import PPO, Buffer

env_fn = lambda: gym.make('Custom_Bipedal-v0')
import numpy as np

env = gym.make('Custom_Bipedal-v0')

print(env.action_space)
print(env.observation_space)

MAX_TIMESTEPS = 200
batch_size = 50
mem = Buffer()
NUM_EPISODES = 5
GAMMA = 0.9

model = PPO()
model.create_model(n_features=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
for episode in range(NUM_EPISODES):
    print(episode)
    obs = env.reset()
    mem.reset()
    for t in range(MAX_TIMESTEPS):
        env.render()
        action, logp = model.sample_action(obs[np.newaxis, :])
        obs, reward, done, _ = env.step(action.numpy().squeeze())  # take a random action

        mem.push(action, obs[np.newaxis, :], logp, reward, done)

