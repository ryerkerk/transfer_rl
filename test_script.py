    import gym
    import transfer_rl.my_env
    from transfer_rl.models import ppo
    env_fn = lambda: gym.make('Custom_Bipedal-v0')
    import numpy as np

    env = gym.make('Custom_Bipedal-v0')

    print(env.action_space)
    print(env.observation_space)

    obs = env.reset()
    model = ppo(n_features=env.observation_space.shape[0], n_actions = env.action_space.shape[0])

    for _ in range(5):
        env.render()
        action = model.get_action(obs[np.newaxis,:], 0)
        obs, reward, done, _ = env.step(action.numpy().squeeze()) # take a random action

    env.close()