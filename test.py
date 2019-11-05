import gym
import gym_simulator
env = gym.make('simulator-v0')
import numpy as np
import matplotlib.pyplot as plt

env.reset()
list = []
for i in range(10000):
     obs, reward, done, info = env.step(0)
     list.append(obs[2]/10000)

plt.plot(range(10000), list)
plt.show()
