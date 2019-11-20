import random

import numpy as np
from keras.layers import Dense, GRU, Masking
from keras.models import Sequential
from keras.optimizers import Adam

from Agents.drqn import ReplayBuffer


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


class DRQNAgent(object):
    def __init__(self,
                 state_size,
                 action_size,
                 lookback=1,
                 batch_size=64,
                 initial_exploration_eps=1000,
                 exploration=LinearSchedule(10000, 0.001)):
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size=int(1e5), replay_batch_size=batch_size, seed=0)
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = np.asarray([0.01 * n for n in range(10)])
        self.lookback = lookback
        self.exploration=exploration

        # These are hyper parameters
        self.discount_factor = 0.99
        self.critic_lr = 1e-3
        self.initial_exploration_eps = initial_exploration_eps

        # create model for Q network
        self.model = self.initialize_model()
        self.target_model = self.initialize_model()
        # initialize target model
        self.update_target_model()

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
            obs_seq = obs_seq.reshape((1, self.lookback, self.state_size))
            q_value = self.model.predict(obs_seq, batch_size=1).flatten()
            s = np.exp(q_value)
            probability_list = s / np.sum(s)
            ac = np.random.choice(self.action_space, 1, p=probability_list)[0]
        else:
            obs_seq = obs_seq.reshape((1, self.lookback, self.state_size))
            q_value = self.model.predict(obs_seq, batch_size=1).flatten()
            idx = np.argmax(q_value)
            ac = self.action_space[idx]
        if random.random() < 0.001:
            print(q_value)
        return ac

    def sample_transition_pairs(self, env, max_step=100):
        obs_s, obs_seq_s, ac_s, rew_s, done_s = [], [], [], [], []
        init_obs = env.reset()
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
            new_obs, rew, done, _ = env.step(ac)
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
        # print(len(self.replay_buffer), self.initial_exploration_eps)
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
                target[i][int(action[i] * 100)] = reward[i]  # TODO: Action dict!
            else:
                target[i][int(action[i] * 100)] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


