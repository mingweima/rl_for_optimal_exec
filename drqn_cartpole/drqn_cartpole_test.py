import datetime
import keras
import gym

from tools.plot_tool import plot_with_avg_std
from drqn_cartpole.drqn_cartpole_train import DRQN_Cartpole_Agent, LinearSchedule


if __name__ == "__main__":

    EPISODES = 10

    # In case of CartPole-v0, maximum length of episode is 200
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = keras.models.load_model('/Users/mmw/Documents/GitHub/rl_for_optimal_exec/drqn_cartpole/drqn_cartpole_v0_10000_eps.h5')
    agent = DRQN_Cartpole_Agent(state_size,
                                action_size,
                                lookback=5,
                                initial_exploration_eps=0,
                                exploration=LinearSchedule(1, 0, initial_p=0),
                                model=model)

    scores, episodes = [], []
    avg_step = 1
    for eps in range(EPISODES):
        eps_rew = agent.sample_transition_pairs(env, render=(eps % avg_step == 0), max_step=500)
        scores.append(eps_rew)
        if eps % avg_step == 0:
            avg = sum(scores[-avg_step-1:-1]) / avg_step
            print('{} episode: {}/{}, average reward: {}'.
                  format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), eps, EPISODES, avg))
        env.reset()

    plot_with_avg_std(scores, 1, xlabel=f'Number of Episodes in {1}')