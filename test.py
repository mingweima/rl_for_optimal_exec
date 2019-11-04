import gym
import gym_simulator
env = gym.make('simulator-v0')

env.reset()
for i in range(10):
    env.step(30)
    env.render()
