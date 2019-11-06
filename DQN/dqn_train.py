import gym
import gym_trading
import numpy as np
import matplotlib.pyplot as plt
from DQN.DQNAgent import DQNAgent

env = gym.make('hwenv-v0')

action_size = 10
state_size = 4
episodes = 2000

agent = DQNAgent(state_size, action_size)

batch_size = 10
rewards_list = []

for i in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    total_reward = 0
    for time_t in range(50):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action/10)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print('episode: {}/{}, total reward: {}, total time: {}'.format(i, episodes, total_reward, time_t))
            break
    rewards_list.append(total_reward)
    agent.replay(10)

plt.plot(range(episodes), rewards_list)
plt.show()