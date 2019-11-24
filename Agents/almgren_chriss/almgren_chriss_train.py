import gym
import gym_trading
import numpy as np
import matplotlib.pyplot as plt
from Agents.almgren_chriss.almgren_chriss_agent import AlmgrenChrissAgent
from tools.plot_tool import plot_with_avg_std

env = gym.make('hwenv-v0')

episodes = 1

agent = AlmgrenChrissAgent(env, time_horizon=env.time_horizon, eta=2.5e-6, rho=0, sigma=1e-3, tau=1, lamb=0)

rewards_list = []

for i in range(episodes):
    agent.reset()
    state = env.reset()
    state = np.reshape(state, [1, 4])
    total_shortfall = 0
    total_time = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_shortfall += info['shortfall']
        # print("reward:", reward)
        state = next_state
        total_time += 1
        if done:
            print('episode: {}/{}, total reward: {}, total time: {}'.format(i, episodes, total_shortfall, total_time))
            break
    env.render()
    rewards_list.append(total_shortfall)


plot_with_avg_std(rewards_list, 1, show=True)
plt.show()
