import random

import tensorflow as tf
import numpy as np
from keras.layers import Dense, GRU, Masking, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam


from Agents.drqn.replay_buffer import ReplayBuffer

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


class DuelingDQNNet:
    def __init__(self, lookback, ob_dim, ac_dim, initial_learning_rate, name):

        self.learning_rate = initial_learning_rate
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.lookback = lookback
        self.name = name

        with tf.variable_scope(self.name):
            self.ob_seqs = tf.placeholder(tf.float32, [None, self.lookback, self.ob_dim], name="ob")
            self.target_Q = tf.placeholder(tf.float32, [None, self.ac_dim], name="target")

            self.output = Masking(mask_value=0., input_shape=(self.lookback, self.ob_dim))(self.ob_seqs)
            self.output = GRU(64, input_dim=(self.lookback, self.ob_dim), kernel_initializer='zeros')(self.output)
            self.output = LeakyReLU(alpha=0.01)(self.output)
            self.output = Dense(64, activation='Linear')(self.output)

            self.value = Dense(1)(self.output)
            self.adv = Dense(self.ac_dim)(self.output)

            self.Q = self.value + tf.subtract(self.adv, tf.reduce_mean(self.adv, axis=1, keepdims=True))

            self.loss = tf.reduce_mean(tf.squared_difference(self.target_Q, self.Q))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def record_tensorboard(self):
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.histogram("target_Q", self.target_Q)
        tf.summary.histogram("value", self.value)
        tf.summary.scalar("loss", self.loss)
        self.merge_opt = tf.summary.merge_all()

        return self.merge_opt

    def get_graph(self):
        return tf.get_default_graph()

class DRQNAgent(object):
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 lookback,
                 batch_size,
                 initial_exploration_steps,
                 exploration=LinearSchedule(10000, 0.001),
                 double='True',
                 dueling='True'):
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size=int(1e5), replay_batch_size=batch_size, seed=0)
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.lookback = lookback
        self.exploration = exploration
        self.double = double
        self.dueling = dueling

        # These are hyper parameters
        self.discount_factor = 1.0
        self.critic_lr = 1e-4
        self.initial_exploration_steps = initial_exploration_steps

        # create model for Q network
        self.model = self.initialize_model('Model')
        self.target_model = self.initialize_model('Target Model')
        self.update_target_model()

        # Number of finished episodes
        self.t = 0

    def initialize_model(self, name=None):
        """
        Approximate Q function using Neural Network:
        obs_seq is input and Q Value of each action is output the of network
        """
        if self.dueling == 'True':
            return DuelingDQNNet(self.lookback, self.ob_dim, self.ac_dim, self.critic_lr, name)
        else:
            model = Sequential()
            model.add(Masking(mask_value=0., input_shape=(self.lookback, self.ob_dim)))
            model.add(GRU(64, input_dim=(self.lookback, self.ob_dim), kernel_initializer='zeros'))
            model.add(LeakyReLU(alpha=0.01))
            model.add(Dense(64, kernel_initializer='he_uniform'))
            model.add(LeakyReLU(alpha=0.01))
            model.add(Dense(self.ac_dim, activation='linear', kernel_initializer='he_uniform'))
            model.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))
        return model

    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, obs_seq):
        """
        Args:
            obs_seq: observation sequence
        Returns:
            ac: integer from 0 to (ac_dim - 1)
        """
        if len(self.replay_buffer) < self.initial_exploration_steps:
            return np.random.choice(self.ac_dim, 1)[0]
        elif random.random() < self.exploration.value(self.t):
            obs_seq = obs_seq.reshape((1, self.lookback, self.ob_dim))
            q_value = self.model.predict(obs_seq, batch_size=1).flatten()
            s = np.exp(q_value)
            probability_list = s / np.sum(s)
            ac = np.random.choice(self.ac_dim, 1, p=probability_list)[0]
        else:
            obs_seq = obs_seq.reshape((1, self.lookback, self.ob_dim))
            q_value = self.model.predict(obs_seq, batch_size=1).flatten()
            ac = np.argmax(q_value)
        return ac

    def sample_trajectory(self, env):
        """
        Sample a trajectory by running the experiment, store all the transition pairs in the replay buffer
        """
        obs_s = []
        init_obs = env.reset()
        obs_s.append(init_obs)
        total_re = 0

        def padding(seq):
            if len(seq) < self.lookback:
                len_to_pad = self.lookback - len(seq)
                pad = [np.zeros_like(init_obs)] * len_to_pad
                seq = pad + seq
            return seq

        while True:
            obs_seq = np.asarray(padding(obs_s[-self.lookback:]))
            ac = self.get_action(obs_seq)
            new_obs, re, done, _ = env.step(ac)
            obs_s.append(new_obs)
            new_obs_seq = np.asarray(padding(obs_s[-self.lookback:]))
            total_re += re
            self.replay_buffer.add(obs_seq, ac, re, new_obs_seq, done)
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

        update_input = np.zeros((batch_size, self.lookback, self.ob_dim))
        update_target = np.zeros((batch_size, self.lookback, self.ob_dim))
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
                target[i][acs[i]] = res[i]
            else:
                if self.double == 'False':
                    target[i][acs[i]] = res[i] + self.discount_factor * (np.amax(target_val[i]))
                elif self.double == 'True':
                    target[i][acs[i]] = res[i] + self.discount_factor * target_val[i][np.argmax(target[i])]
                else:
                    raise Exception('Unknown Double!')

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)