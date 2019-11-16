import numpy as np
import tensorflow as tf

class PGAgent(object):
    def __init__(self, state_dim, action_dim, sample_trajectory_args, estimate_return_args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = 5e-3
        self.gamma = 1
        self.max_path_length = 50
        self.min_timesteps_per_batch = 300
        self.nn_baseline = False

    def build_mlp(input_placeholder, output_size, scope, activation=tf.tanh, output_activation=None):
        with tf.variable_scope(scope):
            layer = input_placeholder
            for _ in range(2):
                layer = tf.layers.dense(layer, 64, activation=activation)
            output_placeholder = tf.layers.dense(layer, output_size, activation=output_activation)
        return output_placeholder

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        tf.global_variables_initializer().run()

    def build_computation_graph(self):
        self.sy_ob_no = tf.placeholder(shape=[None, self.state_dim], name="ob", dtype=tf.float32)
        self.sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

        self.policy_parameters = self.build_mlp(self.sy_ob_no, self.action_dim, "discrete")# dim: (batch_size,)
        self.sy_sampled_ac = tf.reshape(tf.multinomial(self.policy_parameters, 1), [-1])    # dim: (batch_size,)
        self.sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sy_ac_na,
                                                                            logits=self.policy_parameters)
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
            paths.append(path)
            timesteps_this_batch += len(path["reward"])
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
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
