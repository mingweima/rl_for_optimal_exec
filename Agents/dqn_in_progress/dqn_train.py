import datetime
import numpy as np
import gym
import gym_trading

from Agents.dqn.dqn_agent import DQNAgent
from tools.plot_tool import plot_with_avg_std

env = gym.make('hwenv-v0')

action_size = 10
state_size = 4
episodes = 20000

agent = DQNAgent(state_size, action_size)

batch_size = 64
rewards_list = []

for i in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    total_reward = 0
    for time_t in range(60):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action/10)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            if i % 10 == 0:
                print('{} episode: {}/{}, total reward: {}, total time: {}'.
                      format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, episodes, total_reward, time_t))
            break
    rewards_list.append(total_reward)
    agent.replay(batch_size)


plot_with_avg_std(rewards_list, 100, xlabel=f'Number of Episodes in {100}')
