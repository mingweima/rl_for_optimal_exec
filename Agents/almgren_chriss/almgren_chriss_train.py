import gym
import numpy as np
import matplotlib.pyplot as plt
from Agents.almgren_chriss.almgren_chriss_agent import AlmgrenChrissAgent
from tools.plot_tool import plot_with_avg_std

env = gym.make('hwenv-v0')

episodes = 10

agent = AlmgrenChrissAgent(time_horizon=60)

rewards_list = []

for i in range(episodes):
    agent.reset()
    state = env.reset()
    state = np.reshape(state, [1, 4])
    total_reward = 0
    for time_t in range(60):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward
        # print("reward:", reward)
        state = next_state
        print(state[0][1])
        if done:
            print('episode: {}/{}, total reward: {}, total time: {}'.format(i, episodes, total_reward, time_t))
            break
    rewards_list.append(total_reward)


plot_with_avg_std(rewards_list, 1, show=True)
plt.show()
