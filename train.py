import gym
import gym_trading
from collections import deque
import random
import numpy as np
from DQNAgent import DQNAgent

env = gym.make('hwenv-v0')

action_size = 1
state_size = 4
episodes = 100

agent = DQNAgent(state_size, action_size)
for i in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    total_reward = 0
    for time_t in range(50):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print('episode: {}/{}, total reward: {}, total time: {}'.format(i, episodes, total_reward, time_t))
            break
    agent.replay(10)
