import random
from collections import deque

import numpy as np
import keras

from Agents.dqn.replay_buffer import ReplayBuffer

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

class DQNAgent(object):
    """
    The agent that apply the deep Q-learning neural network learning algorithm.
    """
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 batch_size,
                 initial_exploration_steps,
                 exploration=LinearSchedule(10000, 0.001)):
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size=int(1e5), replay_batch_size=batch_size, seed=0)
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.exploration = exploration

        # These are hyper parameters
        self.discount_factor = 0.99
        self.critic_lr = 1e-3
        self.initial_exploration_steps = initial_exploration_steps

        # create model for Q network
        self.model = self.initialize_model()
        self.target_model = self.initialize_model()
        self.update_target_model()

        # Number of finished episodes
        self.t = 0

    def initialize_model(self):
        """
        Build a feedforward neural network.
        """
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim=self.ob_dim, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(self.ac_dim, activation='linear'))
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(lr=self.critic_lr))
        return model

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, ob):
        """
        Args:
            obs_seq: observation sequence
        Returns:
            ac: integer from 0 to (ac_dim - 1)
        """
        if len(self.replay_buffer) < self.initial_exploration_steps:
            return np.random.choice(self.ac_dim, 1)[0]
        elif random.random() < self.exploration.value(self.t):
            q_value = self.model.predict(np.reshape(ob, [1, self.ob_dim]), batch_size=1).flatten()
            s = np.exp(q_value)
            probability_list = s / np.sum(s)
            ac = np.random.choice(self.ac_dim, 1, p=probability_list)[0]
        else:
            q_value = self.model.predict(np.reshape(ob, [1, self.ob_dim]), batch_size=1).flatten()
            ac = np.argmax(q_value)
        return ac

    def sample_trajectory(self, env):
        """
        Sample a trajectory by running the experiment, store all the transition pairs in the replay buffer
        """
        ob = env.reset()
        total_re = 0

        while True:
            ac = self.get_action(ob)
            new_ob, re, done, _ = env.step(ac)
            total_re += re
            self.replay_buffer.add(ob, ac, re, new_ob, done)
            if done:
                break
        self.t += 1
        return total_re

    def train_model(self):
        """
        Train the model.
        """
        if len(self.replay_buffer) < self.initial_exploration_steps:
            return
        batch_size = min(self.batch_size, len(self.replay_buffer))
        mini_batch = [self.replay_buffer.sample() for _ in range(batch_size)]

        update_input = np.zeros((batch_size, self.ob_dim))
        update_target = np.zeros((batch_size, self.ob_dim))
        acs, res, dones = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            acs.append(mini_batch[i][1])
            res.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if dones[i]:
                target[i][acs[i]] = res[i]  # TODO: Action dict!
            else:
                    target[i][acs[i]] = res[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)