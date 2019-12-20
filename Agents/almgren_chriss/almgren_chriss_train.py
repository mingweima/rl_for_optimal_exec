import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_trading

from Agents.almgren_chriss.almgren_chriss_agent import AlmgrenChrissAgent
from tools.plot_tool import plot_with_avg_std

def AlmgrenChrissTrain(scenario_args, observation_space_args, action_space_args, reward_args):

    EPISODES = 1

    env = gym.make('hwenv-v0',
                   scenario_args=scenario_args,
                   observation_space_args=observation_space_args,
                   action_space_args=action_space_args,
                   reward_args=reward_args)

    agent = AlmgrenChrissAgent(env, time_horizon=env.time_horizon, eta=2.5e-6, rho=0, sigma=1e-3, tau=1, lamb=0)
    ob_dim = env.observation_space.shape[0]

    rewards_list = []

    for i in range(EPISODES):
        agent.reset()
        state = env.reset()
        state = np.reshape(state, [1, ob_dim])
        total_shortfall = 0
        total_time = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, ob_dim])
            total_shortfall += info['shortfall']
            state = next_state
            total_time += 1
            if done:
                print(
                    'episode: {}/{}, total reward: {}, total time: {}'.format(i, EPISODES, total_shortfall, total_time))
                break
        rewards_list.append(total_shortfall)

    plot_with_avg_std(rewards_list, 1, show=True)
    plt.show()