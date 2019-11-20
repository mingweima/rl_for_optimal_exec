import numpy as np
import tensorflow as tf

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

    def build_mlp(self, input_placeholder, output_size, scope):
        with tf.variable_scope(scope):
            layer = input_placeholder
            for _ in range(2):
                layer = tf.layers.dense(layer, 64, activation=tf.tanh)
            output_placeholder = tf.layers.dense(layer, output_size, activation=None)
        return output_placeholder

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        tf.global_variables_initializer().run()

    def build_computation_graph(self):
        self.sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        self.sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

        # self.policy_parameters: dim: (batch_size, num_of_actions)
        # the prediction from the neural network given an observation
        self.policy_parameters = self.build_mlp(self.sy_ob_no, self.ac_dim, "discrete")
        self.sy_sampled_ac = tf.reshape(tf.multinomial(self.policy_parameters, 1), [-1])

        # self.sy_log_pro_n: log(pi(a|s))
        # this is a function of the difference between f(s) and a, for discrete case, log(pi(a|s)) is given
        # by the softmax cross entropy, while for continuous case, given by the Gaussian Distribution
        self.sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sy_ac_na,
                                                                            logits=self.policy_parameters)

        # loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
        # to get the policy gradient.
        loss = - tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        if self.nn_baseline:
            self.baseline_prediction = tf.squeeze(self.build_mlp(self.sy_ob_no, 1, "nn_baseline",))
            self.sy_target_n = tf.placeholder(shape=[None], name="baseline", dtype=tf.float32)
            baseline_loss = tf.nn.l2_loss(self.baseline_prediction - self.sy_target_n)
            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(baseline_loss)

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
            if animate_this_episode:
                env.render()
            obs.append(ob)
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: [ob]})
            # ac looks like: [x], ac[0] looks like: x
            ac = ac[0]
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

    def compute_advantage(self, ob_no, q_n):
        # Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        # ob_no: shape: (sum_of_path_lengths, ob_dim)
        # q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values.
        # Output: shape: (sum_of_path_lengths). A single vector for the estimated advantages.
        if self.nn_baseline:
            b_n = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no})
            b_n = (b_n - np.mean(b_n)) / np.std(b_n)
            b_n = b_n * np.std(q_n) + np.mean(q_n)
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, re_n):
        # Estimates the returns over a set of trajectories.
        # returns:
        #     q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
        #         whose length is the sum of the lengths of the paths
        #     adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
        #         advantages whose length is the sum of the lengths of the paths
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n, axis=0)) / np.std(adv_n, axis=0)
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n):
        """
            Update the parameters of the policy and (possibly) the neural network baseline,
            which is trained to approximate the value function.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        if self.nn_baseline:
            target_n = (q_n - np.mean(q_n)) / np.std(q_n)
            self.sess.run(self.baseline_update_op, feed_dict={self.sy_ob_no: ob_no, self.sy_target_n: target_n})
        self.sess.run(self.update_op, feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_adv_n: adv_n})