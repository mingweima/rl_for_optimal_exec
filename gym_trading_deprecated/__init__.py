from gym.envs.registration import register

register(
    id='hwenv-v0',
    entry_point='gym_trading_deprecated.hw_sim:Simulator',
)