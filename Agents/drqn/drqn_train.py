import datetime
import gym
import gym_trading
import matplotlib.pyplot as plt
from keras.models import load_model

from Agents.drqn.drqn_agent import DRQNAgent
from tools.plot_tool import plot_with_avg_std


def DRQNTrain(scenario_args, observation_space_args,
              action_space_args, reward_args, data_args, almgren_chriss_args, double):

    EPISODES = 30000

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

    agent = DRQNAgent(ob_dim,
                      ac_dim,
                      lookback=30,
                      batch_size=256,
                      initial_exploration_steps=1000,
                      double=double)

    scores = []
    avgs = []
    avg_step = 100
    for eps in range(EPISODES):
        eps_rew = agent.sample_trajectory(env)
        scores.append(eps_rew)
        if (eps % avg_step) == 0 & (eps != 0):
            avg = sum(scores[-avg_step-1:-1]) / avg_step
            avgs.append(avg)
            print('{} episode: {}/{}, average reward: {}'.
                  format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), eps, EPISODES, avg))
            # env.render()
        agent.train_model()
        if eps % 5 == 0:
            agent.update_target_model()

    agent.target_model.save('model.h5')
    print('Saved model to disk.')

    plot_with_avg_std(avgs, 10)

def DRQNTest():
    # load model
    model = load_model('model.h5')
    # summarize model
    model.summary()