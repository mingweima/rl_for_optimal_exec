import gym
import gym_trading
import numpy as np
import matplotlib.pyplot as plt
from Agents.almgren_chriss.almgren_chriss_agent import AlmgrenChrissAgent
from tools.plot_tool import plot_with_avg_std


class LinearAgent(object):
    def __init__(self, steps=10):
        self.steps = steps
        self.done_step = 0

    def act(self):
        percentage_left = 1 - (self.done_step/self.steps)
        percentage_of_current_to_act = (1/self.steps) / percentage_left
        self.done_step += 1
        return int(10*percentage_of_current_to_act)

    def reset(self):
        self.done_step = 0


env = gym.make('hwenv-v0')

episodes = 20
agent = LinearAgent(steps=22)


shortfall_list = []
actions_list = []

for i in range(episodes):
    _ = env.reset()
    total_shortfall = 0
    for time_t in range(60):
        action = agent.act()
        _, _, done, info = env.step(action)
        total_shortfall += info['shortfall']
        actions_list.append(action)
        if done:
            print('episode: {}/{}, total shortfall: {}, total time: {}'.format(i, episodes, total_shortfall, time_t))
            env.render()
            actions_list = []
            agent.reset()
            break
    shortfall_list.append(total_shortfall)


plot_with_avg_std(shortfall_list, 1, show=True)
plt.show()


shortfall_list = []