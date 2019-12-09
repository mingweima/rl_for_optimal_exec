import datetime
import gym
import gym_trading

from Agents.drqn.drqn_agent import DRQNAgent
from tools.plot_tool import plot_with_avg_std


if __name__ == "__main__":

    EPISODES = 100000

    env = gym.make('hwenv-v0')
    # get size of state and action from environment
    state_size = 4
    action_size = 10

    agent = DRQNAgent(state_size, action_size, lookback=5, batch_size=64, initial_exploration_eps=5000)

    scores, episodes = [], []
    avg_step = 100
    for eps in range(EPISODES):
        eps_rew = agent.sample_transition_pairs(env, max_step=70)
        scores.append(eps_rew)
        if eps % avg_step == 0:
            avg = sum(scores[-avg_step-1:-1]) / avg_step
            print('{} episode: {}/{}, average reward: {}'.
                  format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), eps, EPISODES, avg))
        agent.train_model()
        if eps % 10 == 0:
            agent.update_target_model()

    plot_with_avg_std(scores, 500, xlabel=f'Number of Episodes in {500}')
