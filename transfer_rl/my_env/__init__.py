from gym.envs.registration import register

register(
    id='Custom_Bipedal-v0',
    entry_point='transfer_rl.my_env.bipedal_walker_test:CustomBipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)