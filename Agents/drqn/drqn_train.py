import datetime
import gym
import gym_trading

from Agents.drqn.drqn_agent import DRQNAgent
from tools.plot_tool import plot_with_avg_std


def DRQNTrain(scenario_args, observation_space_args,
              action_space_args, reward_args, data_args, almgren_chriss_args):

    EPISODES = 100000

    env = gym.make('hwenv-v0',
                   scenario_args=scenario_args,
                   observation_space_args=observation_space_args,
                   action_space_args=action_space_args,
                   reward_args=reward_args,
                   data_args=data_args,
                   almgren_chriss_args=almgren_chriss_args)

    # get size of state and action from environment
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n

    agent = DRQNAgent(ob_dim, ac_dim, lookback=5, batch_size=64, initial_exploration_steps=10000)

    scores = []
    avg_step = 10
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

    plot_with_avg_std(scores, 10, xlabel=f'Number of Episodes in {10}')
