import gym
import numpy as np
from keras import layers
from keras.models import Sequential
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers

class PGAgent(object):
    def __init__(self, ob_dim, ac_dim):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.learning_rate = 5e-3
        self.gamma = 1
        self.max_path_length = 1000
        self.min_timesteps_per_batch = 2000
        self.nn_baseline = True
        self.reward_to_go = True
        self.normalize_advantages = True
        self.model = self.__build_network(ob_dim, ac_dim)
        self.__build_train_fn()

    def __build_network(self, ob_dim, ac_dim):
        model = Sequential()
        model.add(layers.Dense(64, input_dim=self.ob_dim, activation='tanh'))
        model.add(layers.Dense(64, activation='tanh'))
        model.add(layers.Dense(self.ac_dim, activation='softmax'))
        return model

    def __build_train_fn(self):
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.ac_dim), name="action_onehot")
        adv_n = K.placeholder(shape=(None,), name="advantage")
        log_action_prob = K.log(K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1))
        loss = - K.mean(log_action_prob * adv_n)
        adam = optimizers.Adam(learning_rate=self.learning_rate)
        updates = adam.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           adv_n],
                                   outputs=[],
                                   updates=updates)

    def sample_trajectories(self, itr, env):
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode = (len(paths)==0 and (itr % 10 == 0))
            path = self.sample_trajectory(env, animate_this_episode)
            # print(len(path["reward"]))
            paths.append(path)
            timesteps_this_batch += len(path["reward"])
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        print(timesteps_this_batch/len(paths))
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            # if animate_this_episode:
            #     env.render()
            obs.append(ob)
            action_prob = np.squeeze(self.model.predict(np.reshape(ob, [1, 4])))
            ac = np.random.choice(np.arange(self.ac_dim), p=action_prob)
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation": np.array(obs, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(acs, dtype=np.float32)}
        return path

    def sum_of_rewards(self, re_n):
        # Monte Carlo Estimation of the Q function
        # Input: Each element in re_n is a numpy array containing the rewards of a particular path
        # Output: A single vector of Q values whose length is the sum of the lengths of the paths in re_n
        q_n = []
        if self.reward_to_go:
            for re in re_n:
                for t in range(len(re)):
                    gamma_runner = 1
                    q = 0
                    for i in range(len(re) - t):
                        q += re[i + t] * gamma_runner
                        gamma_runner *= self.gamma
                    q_n.extend([q])
        else:
            for re in re_n:
                q = 0
                gamma_runner = 1
                for r in re:
                    q += r * gamma_runner
                    gamma_runner *= self.gamma
                q_n.extend([q] * len(re))
        return q_n

    def estimate_return(self, ob_no, re_n):
        adv_n = self.sum_of_rewards(re_n)
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n, axis=0)) / np.std(adv_n, axis=0)
        return adv_n

    def update_parameters(self, ob_no, ac_na, adv_n):
        action_onehot = np_utils.to_categorical(ac_na, num_classes=self.ac_dim)
        self.train_fn([ob_no, action_onehot, adv_n])

