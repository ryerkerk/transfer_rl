import gym
import my_env

# env = gym.make('BipedalWalker-v2')
env = gym.make('Custom_Bipedal-v0')
# env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

env.close()