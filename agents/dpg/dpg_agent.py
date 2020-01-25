import numpy as np
from keras import layers
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras.models import Sequential

class DPGAgent(object):
    """
    The agent that applies the deep policy gradient learning algorithm.

        Attributes:
            ob_dim (int32): the dimension of the observation space
            ac_dim (int32): the number of different actions available
            learning_rate (float64): the learning rate to update the neural network
            gamma (float64): the discount factor used when computing the Q-values
            min_timesteps_per_batch (int32): the least number of time steps within each batch
            reward_to_go (boolean): whether to calculate Q-values as rewards to go or total rewards
    """

    def __init__(self, ob_dim, ac_dim):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.learning_rate = 5e-3
        self.gamma = 1
        self.min_timesteps_per_batch = 200
        self.reward_to_go = True

        self.model = self.__build_network(ob_dim, ac_dim)
        self.__build_train_fn()

    def __build_network(self, ob_dim, ac_dim):
        """
        Build a feedforward neural network.
        """
        model = Sequential()
        model.add(layers.Dense(64, input_dim=self.ob_dim, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.ac_dim, activation='softmax'))
        return model

    def __build_train_fn(self):
        """
            Define self.train_fn as the function to update the neural network.
        """
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

    def sample_trajectories(self, itr, env, info_name=None, render = False):
        """
            Collect paths until we have enough time steps in the batch.

            Args:
                itr (int32): the itr's batch being sampled
                env: the gym trading_environment
                info_name (string): the name of the info being processed
                render (boolean): render or not
            Returns:
                paths (list): a list dictionaries (each a path) in this batch.
                timesteps_this_batch (int32): the total number of timesteps in the batch.
                avg_rew (float64): the average total rewards
                avg_info (float64): the average total infos
        """
        timesteps_this_batch = 0
        paths = []
        total_rew = 0
        total_info, avg_info = 0, 0
        while True:
            animate_this_episode = (len(paths)==0 and (itr % 10 == 0) and render)
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
        """
            Sample a trajectory with the given policy.

            Args:
                env: the gym trading_environment
                animate_this_episode (boolean): An indication of whether to animate this episode.
            Returns:
                path (dict): a dictionary of the sampled trajectory, with keys being
                    "observation", "reward", "action", and "info".
        """
        ob = env.reset()
        obs, acs, rewards, infos = [], [], [], []
        steps = 0
        while True:
            obs.append(ob)
            action_prob = np.squeeze(self.model.predict(np.reshape(ob, [1, self.ob_dim])))
            ac = np.random.choice(np.arange(self.ac_dim), p=action_prob)
            acs.append(ac)
            ob, rew, done, info = env.step(ac)
            infos.append(info)
            rewards.append(rew)
            steps += 1
            if done:
                break
        path = {"observation": np.array(obs, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(acs, dtype=np.float32),
                "info": infos}
        if animate_this_episode:
            env.render()
        return path

    def sum_of_rewards(self, re_n):
        """
            Monte Carlo estimation of the Q function.

            Let sum_of_path_lengths be the sum of the lengths of the paths sampled
                from DPGAgent.sample_trajectories
            Let num_paths be the number of paths sampled from DPGAgent.sample_trajectories

            Args:
                re_n (ndarray): length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path
            Returns:
                q_n (list): shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
        """
        q_n = []
        if self.reward_to_go:
            # estimate q values by the discounted sum of rewards starting from time step t
            for re in re_n:
                for t in range(len(re)):
                    gamma_runner = 1
                    q = 0
                    for i in range(len(re) - t):
                        q += re[i + t] * gamma_runner
                        gamma_runner *= self.gamma
                    q_n.extend([q])
        else:
            # estimate q values as the total discounted reward summed over the entire trajectory
            for re in re_n:
                q = 0
                gamma_runner = 1
                for r in re:
                    q += r * gamma_runner
                    gamma_runner *= self.gamma
                q_n.extend([q] * len(re))
        return q_n

    def estimate_return(self, re_n):
        """
            Estimates the returns over a set of trajectories.

            Let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                DPGAgent.sample_trajectories
            Let num_paths be the number of paths sampled from DPGAgent.sample_trajectories

            Args:
                re_n (list): length: num_paths. Each element in re_n is a numpy array
                    containing the rewards for the particular path
            Returns:
                adv_n (list): shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        adv_n = self.sum_of_rewards(re_n)
        adv_n = (adv_n - np.mean(adv_n, axis=0)) / np.std(adv_n, axis=0)
        return adv_n

    def update_parameters(self, ob_no, ac_na, adv_n):
        """
            Update the parameters of the neural network.

            Args:
                ob_no (ndarray): shape: (sum_of_path_lengths, ob_dim)
                ac_na (ndarray): shape: (sum_of_path_lengths).
                adv_n (ndarray): shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
            Returns:
                nothing
        """
        action_onehot = np_utils.to_categorical(ac_na, num_classes=self.ac_dim)
        self.train_fn([ob_no, action_onehot, adv_n])