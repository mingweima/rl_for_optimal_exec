from gym.envs.registration import register

register(
    id='hwenv-v0',
    entry_point='gym_trading.hw_sim:Simulator',
)