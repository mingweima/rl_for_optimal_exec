import numpy as np

from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers


class ACAgent(object):
    def __init__(self, ob_dim, ac_dim):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.learning_rate = 5e-3
        self.gamma = 1.0
        self.min_timesteps_per_batch = 200
        self.normalize_advantages = True
        self.num_grad_steps_per_target_update = 10
        self.num_target_updates = 10

        self.actor_model = self.initialize_actor_model()
        self.__build_train_fn()
        self.critic_model = self.initialize_critic_model()
        self.num_sampled_trajectories = 0

    def initialize_actor_model(self):
        model = Sequential()
        model.add(layers.Dense(64, input_dim=self.ob_dim, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.ac_dim, activation='softmax'))
        return model

    def initialize_critic_model(self):
        model = Sequential()
        model.add(layers.Dense(64, input_dim=self.ob_dim, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def sample_trajectories(self, itr, env, info_name=None):
        timesteps_this_batch = 0
        paths = []
        total_rew = 0
        total_info, avg_info = 0, 0
        while True:
            animate_this_episode = (len(paths)==0 and (itr % 1 == 0))
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += len(path["reward"])
            total_rew += np.sum(path["reward"])
            if info_name is not None:
                info_dict_list = path['info']
                for dict in info_dict_list:
                    info = dict[info_name]
                    total_info += info
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        avg_info = total_info / len(paths)
        avg_rew = total_rew / len(paths)
        return paths, timesteps_this_batch, avg_rew, avg_info

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards, next_obs, terminals, infos = [], [], [], [], [], []
        steps = 0
        while True:
            obs.append(ob)
            action_prob = np.squeeze(self.actor_model.predict(np.reshape(ob, [1, 4])))
            ac = np.random.choice(np.arange(self.ac_dim), p=action_prob)
            ob, rew, done, info = env.step(ac)
            acs.append(ac)
            infos.append(info)
            next_obs.append(ob)
            rewards.append(rew)
            steps += 1
            if done:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        path = {"observation": np.array(obs, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32),
                "info": infos}
        self.num_sampled_trajectories += 1
        # if animate_this_episode:
        #     env.render()
        return path

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        v_n = self.critic_model.predict(ob_no).flatten()
        next_v_n = self.critic_model.predict(next_ob_no).flatten()
        adv_n = re_n + (1 - terminal_n) * self.gamma * next_v_n - v_n
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n, axis=0)) / np.std(adv_n, axis=0)
        return adv_n

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        n = self.num_grad_steps_per_target_update * self.num_target_updates
        for t in range(0, n):
            if t % self.num_grad_steps_per_target_update == 0:
                next_v_n = self.critic_model.predict(next_ob_no).flatten()
                target_n = re_n + (1 - terminal_n) * self.gamma * next_v_n
            self.critic_model.fit(ob_no, target_n, epochs=1, verbose=0)

    def update_actor(self, ob_no, ac_na, adv_n):
        ac_onehot = np_utils.to_categorical(ac_na, num_classes=self.ac_dim)
        self.train_fn([ob_no, ac_onehot, adv_n])

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

