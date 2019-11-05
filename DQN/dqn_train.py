import gym
import gym_trading
import numpy as np
import matplotlib.pyplot as plt
from DQN.DQNAgent import DQNAgent

env = gym.make('hwenv-v0')

action_size = 1
state_size = env.observation_space.shape[0]
episodes = 2000

agent = DQNAgent(state_size, action_size)

batch_size = 10
rewards = []
episode_list = []

for i in range(episodes):
    state = np.reshape(env.reset(), [1, 4])
    total_reward = 0
    for time_t in range(50):
        action = agent.act(state)
        # print('HI', action)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print('episode: {}/{}, total reward: {}, total time: {}'.format(i, episodes, total_reward, time_t))
            rewards.append(total_reward)
            episode_list.append(i)
            break
        agent.replay(50)

plt.plot(episode_list, rewards)
plt.show()