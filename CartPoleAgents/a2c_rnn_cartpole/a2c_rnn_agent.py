import numpy as np

from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers


class ACRnnAgent(object):
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 action_space=[0, 1],
                 learning_rate=1e-3,
                 gamma=0.99,
                 max_path_length=600,
                 batch_size=1000,
                 num_grad_steps_per_target_update=10,
                 num_target_updates=10,
                 lookback=5,
                 initial_exploration_eps=10
                 ):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.normalize_advantages = True
        self.num_grad_steps_per_target_update = num_grad_steps_per_target_update
        self.num_target_updates = num_target_updates
        self.lookback = lookback
        self.initial_exploration_eps = initial_exploration_eps

        self.actor_model = self.initialize_actor_model()
        self.__build_train_fn()
        self.critic_model = self.initialize_critic_model()
        self.num_sampled_trajectories = 0

    def initialize_actor_model(self):
        model = Sequential()
        model.add(layers.Masking(mask_value=0., input_shape=(self.lookback, self.ob_dim)))
        model.add(layers.GRU(16, input_dim=(self.lookback, self.ob_dim), activation='tanh',
                      kernel_initializer='zeros'))
        model.add(layers.Dense(self.ac_dim, activation='softmax'))
        return model

    def initialize_critic_model(self):
        model = Sequential()
        model.add(layers.Masking(mask_value=0., input_shape=(self.lookback, self.ob_dim)))
        model.add(layers.GRU(16, input_dim=(self.lookback, self.ob_dim), activation='tanh', kernel_initializer='zeros'))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def sample_trajectories(self, env, render=False, animate_eps_frequency=10):
        timesteps_this_batch = 0
        paths = []
        total_rew = 0
        while True:
            path = self.sample_trajectory(env, render=render, animate_eps_frequency=animate_eps_frequency)
            paths.append(path)
            timesteps_this_batch += len(path["reward"])
            total_rew += np.sum(path["reward"])
            if timesteps_this_batch > self.batch_size:
                break
        avg_rew = total_rew / len(paths)
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = np.concatenate([path["reward"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        terminal_n = np.concatenate([path["terminal"] for path in paths])
        ob_seq = np.concatenate([path["observation_sequence"] for path in paths])
        next_ob_seq = np.concatenate([path["next_observation_sequence"] for path in paths])
        return ob_seq, next_ob_seq, ac_na, re_n, terminal_n, avg_rew

    def sample_trajectory(self, env, render=False, animate_eps_frequency=10):
        ob = env.reset()
        obs, obs_seq_s, next_obs_seq_s, acs, rewards, next_obs, terminals = [], [], [], [], [], [], []
        steps = 0

        def padding(seq):
            if len(seq) < self.lookback:
                len_to_pad = self.lookback - len(seq)
                pad = [np.zeros_like(ob)] * len_to_pad
                # pad = [init_obs] * len_to_pad
                seq = pad + seq
            return np.asarray(seq)

        while True:
            obs.append(ob)
            obs_seq = padding(obs[-self.lookback:])
            next_obs_seq_candidate = obs_seq[1 :]
            obs_seq = obs_seq.reshape((self.lookback, self.ob_dim))
            obs_seq_s.append(obs_seq)
            ac = self.get_action(obs_seq)
            ob, rew, done, _ = env.step(ac)
            next_obs_seq = np.concatenate([next_obs_seq_candidate, np.asarray([ob])])
            next_obs_seq = next_obs_seq.reshape((self.lookback, self.ob_dim))
            if render and self.num_sampled_trajectories % animate_eps_frequency == 0:
                env.render()
            acs.append(ac)
            next_obs.append(ob)
            next_obs_seq_s.append(next_obs_seq)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        path = {"observation": np.array(obs, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32),
                "observation_sequence": np.array(obs_seq_s, dtype=np.float32),
                "next_observation_sequence": np.array(next_obs_seq_s, dtype=np.float32)}
        self.num_sampled_trajectories += 1
        return path

    def get_action(self, obs_seq):
        obs_seq = obs_seq.reshape((1, self.lookback, self.ob_dim))
        action_prob = np.squeeze(self.actor_model.predict(obs_seq, batch_size=1))
        ac = np.random.choice(self.action_space, p=action_prob)
        return ac

    def estimate_advantage(self, ob_seq, next_ob_seq, re_n, terminal_n):
        v_n = self.critic_model.predict(ob_seq).flatten()
        next_v_n = self.critic_model.predict(next_ob_seq).flatten()
        adv_n = re_n + (1 - terminal_n) * self.gamma * next_v_n - v_n
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n, axis=0)) / np.std(adv_n, axis=0)
        return adv_n

    def update_critic(self, ob_seq, next_ob_seq, re_n, terminal_n):
        n = self.num_grad_steps_per_target_update * self.num_target_updates
        for t in range(0, n):
            if t % self.num_grad_steps_per_target_update == 0:
                next_v_n = self.critic_model.predict(next_ob_seq).flatten()
                target_n = re_n + (1 - terminal_n) * self.gamma * next_v_n
            self.critic_model.fit(ob_seq, target_n, epochs=1, verbose=0)

    def update_actor(self, ob_seq, ac_na, adv_n):
        ac_onehot = np_utils.to_categorical(ac_na, num_classes=self.ac_dim)
        self.train_fn([ob_seq, ac_onehot, adv_n])

    def __build_train_fn(self):
        action_prob_placeholder = self.actor_model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.ac_dim), name="action_onehot")
        adv_n = K.placeholder(shape=(None,), name="advantage")
        log_action_prob = K.log(K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1))
        loss = - K.mean(log_action_prob * adv_n)
        adam = optimizers.Adam(learning_rate=self.learning_rate)
        updates = adam.get_updates(params=self.actor_model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.actor_model.input,
                                           action_onehot_placeholder,
                                           adv_n],
                                   outputs=[],
                                   updates=updates)

