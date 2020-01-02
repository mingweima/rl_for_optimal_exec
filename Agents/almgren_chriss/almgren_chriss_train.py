import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_trading

from Agents.almgren_chriss.almgren_chriss_agent import AlmgrenChrissAgent
from tools.plot_tool import plot_with_avg_std

def AlmgrenChrissTrain(scenario_args, observation_space_args,
                       action_space_args, reward_args, data_args, almgren_chriss_args):

    EPISODES = 1

    env = gym.make('hwenv-v0',
                   scenario_args=scenario_args,
                   observation_space_args=observation_space_args,
                   action_space_args=action_space_args,
                   reward_args=reward_args,
                   data_args=data_args,
                   almgren_chriss_args=almgren_chriss_args)


    agent = AlmgrenChrissAgent(env,
                               time_horizon=scenario_args['Time Horizon'],
                               eta=almgren_chriss_args['eta'],
                               rho=almgren_chriss_args['rho'],
                               sigma=almgren_chriss_args['sigma'],
                               tau=scenario_args['Trading Interval'],
                               lamb=almgren_chriss_args['lamb'],
                               kappa=almgren_chriss_args['kappa'])
    ob_dim = env.observation_space.shape[0]

    rewards_list = []

    for i in range(EPISODES):
        agent.reset()
        state = env.reset()
        state = np.reshape(state, [1, ob_dim])
        total_shortfall = 0
        total_time = 0
        done = False

        time = []
        size = []
        price = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            time.append(info['time'])
            size.append(info['size'])
            price.append(info['price_before_action'])

            next_state = np.reshape(next_state, [1, ob_dim])
            total_shortfall += info['shortfall']
            state = next_state
            total_time += 1
            if done:
                print('episode: {}/{}, total reward: {}, total steps: {}'.format(i + 1,
                                                                                 EPISODES,
                                                                                 total_shortfall,
                                                                                 total_time))
                break
        rewards_list.append(total_shortfall)

    env.render()

    # plot_with_avg_std(rewards_list, 1, xlabel=f'Number of Episodes in {1}')

    return rewards_list