import datetime
import numpy as np
import gym
import gym_trading

from Agents.dqn.dqn_agent import DQNAgent
from tools.plot_tool import plot_with_avg_std

if __name__ == "__main__":

    EPISODES = 100000

    env = gym.make('hwenv-v0')
    # get size of state and action from environment
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n

    agent = DQNAgent(ob_dim, ac_dim, batch_size=64, initial_exploration_steps=1000)

    scores = []
    avg_step = 100
    for eps in range(EPISODES):
        eps_rew = agent.sample_trajectory(env)
        scores.append(eps_rew)
        if eps % avg_step == 0:
            avg = sum(scores[-avg_step-1:-1]) / avg_step
            print('{} episode: {}/{}, average reward: {}'.
                  format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), eps, EPISODES, avg))
        agent.train_model()
        if eps % 10 == 0:
            agent.update_target_model()

    plot_with_avg_std(scores, 500, xlabel=f'Number of Episodes in {500}')
