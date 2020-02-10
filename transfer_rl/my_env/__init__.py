from gym.envs.registration import register

register(
    id='Bipedal_Custom_Leg_Length-v0',
    entry_point='transfer_rl.my_env.bipedal_custom_leg_length:BipedalWalkerCustomLegLength',
    max_episode_steps=1500,
    reward_threshold=300,
)