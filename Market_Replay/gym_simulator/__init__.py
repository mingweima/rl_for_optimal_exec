from gym.envs.registration import register

register(
    id='simulator-v0',
    entry_point='gym_simulator.envs:Simulator',
)