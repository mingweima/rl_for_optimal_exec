import datetime
import gym

from tools.plot_tool import plot_with_avg_std

import sys
import datetime
import random

import numpy as np
import keras
from keras.layers import Dense, LSTM, GRU, Masking
from keras.models import Sequential
from keras.optimizers import Adam
import gym
import gym_trading
import pylab

from drqn.replay_buffer import ReplayBuffer
from tools.plot_tool import plot_with_avg_std


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class DRQN_Cartpole_Agent(object):
    def __init__(self,
                 state_size,
                 action_size,
                 action_space=np.asarray([0, 1]),
                 lookback=1,
                 batch_size=64,
                 initial_exploration_eps=1000,
                 exploration=LinearSchedule(1000, 0.001),
                 buffer_size=int(1e6),
                 model=None):
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, replay_batch_size=batch_size, seed=0)
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.lookback = lookback
        self.exploration=exploration

        # These are hyper parameters
        self.discount_factor = 0.95
        self.critic_lr = 1e-4
        self.initial_exploration_eps = initial_exploration_eps

        # create model for Q network
        if model is None:
            self.model = self.initialize_model()
            self.target_model = self.initialize_model()
            # initialize target model
            self.update_target_model()
        else:
            self.model = model
            self.target_model = model

        self.t = 0

    # approximate Q function using Neural Network
    # obs_seq is input and Q Value of each action is output of network
    def initialize_model(self):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(self.lookback, self.state_size)))
        model.add(GRU(16, input_dim=(self.lookback, self.state_size), activation='tanh',
                       kernel_initializer='zeros'))
        model.add(Dense(256, activation='linear',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))
        return model

        # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, obs_seq):
        if len(self.replay_buffer) < self.initial_exploration_eps:
            return np.random.choice(self.action_space, 1)[0]
        elif random.random() < self.exploration.value(self.t):
            # obs_seq = obs_seq.reshape((1, self.lookback, self.state_size))
            # q_value = self.model.predict(obs_seq, batch_size=1).flatten()
            # norm_q_value = q_value / np.sum(abs(q_value))
            # s = np.exp(norm_q_value)
            # probability_list = s / np.sum(s)
            # # print('plist', probability_list)
            # ac = np.random.choice(self.action_space, 1, p=probability_list)[0]
            return np.random.choice(self.action_space, 1)[0]
        else:
            obs_seq = obs_seq.reshape((1, self.lookback, self.state_size))
            q_value = self.model.predict(obs_seq, batch_size=1).flatten()
            idx = np.argmax(q_value)
            ac = self.action_space[idx]
        if random.random() < 0.001:
            print(q_value)
        return ac

    def sample_transition_pairs(self, train_env, max_step=100):
        obs_s, obs_seq_s, ac_s, rew_s, done_s = [], [], [], [], []
        init_obs = train_env.reset()
        obs_s.append(init_obs)
        total_rew = 0

        def padding(seq):
            if len(seq) < self.lookback:
                len_to_pad = self.lookback - len(seq)
                pad = [np.zeros_like(init_obs)] * len_to_pad
                # pad = [init_obs] * len_to_pad
                seq = pad + seq
            return seq

        for step in range(max_step):
            obs_seq = obs_s[-self.lookback:]  # a list with max_len = lookback
            obs_seq = padding(obs_seq)
            obs_seq = np.asarray(obs_seq)
            ac = self.get_action(obs_seq)
            # print(ac)
            new_obs, rew, done, _ = train_env.step(ac)
            obs_s.append(new_obs)
            ac_s.append(ac)
            rew_s.append(rew)
            done_s.append(done)
            new_obs_seq = obs_s[-self.lookback:]
            new_obs_seq = padding(new_obs_seq)
            new_obs_seq = np.asarray(new_obs_seq)
            total_rew += rew
            self.replay_buffer.add(obs_seq, ac, rew, new_obs_seq, done)
            if done:
                break
        self.t += 1
        return total_rew

    def train_model(self):
        if len(self.replay_buffer) < self.initial_exploration_eps:
            return

        batch_size = min(self.batch_size, len(self.replay_buffer))
        mini_batch = [self.replay_buffer.sample() for _ in range(batch_size)]

        update_input = np.zeros((batch_size, self.lookback, self.state_size))
        update_target = np.zeros((batch_size, self.lookback, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][int(action[i])] = reward[i]  # TODO: Action dict!
            else:
                target[i][int(action[i])] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":

    EPISODES = 20000

    # In case of CartPole-v0, maximum length of episode is 200
    env = gym.make('CartPole-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(state_size, action_size)

    agent = DRQN_Cartpole_Agent(state_size, action_size, lookback=5, batch_size=64, initial_exploration_eps=1000)

    scores, episodes = [], []
    avg_step = 100
    for eps in range(EPISODES):
        eps_rew = agent.sample_transition_pairs(env, max_step=200)
        scores.append(eps_rew)
        if eps % avg_step == 0:
            avg = sum(scores[-avg_step:-1]) / avg_step
            print('{} episode: {}/{}, average reward: {}'.
                  format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), eps, EPISODES, avg))
        agent.train_model()
        if eps % 1 == 0:
            agent.update_target_model()
        env.reset()
        if eps % 5000 == 0:
            agent.model.save(f'drqn_cartpole_v0_{int(eps)}_eps.h5')

    plot_with_avg_std(scores, 500, xlabel=f'Number of Episodes in {500}')
