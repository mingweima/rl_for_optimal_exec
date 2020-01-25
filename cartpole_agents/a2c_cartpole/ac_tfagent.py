import numpy as np
import tensorflow as tf


class ACAgent(object):
    def __init__(self, ob_dim, ac_dim):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.learning_rate = 5e-3
        self.gamma = 1
        self.max_path_length = 600
        self.min_timesteps_per_batch = 2000
        self.nn_baseline = True
        self.reward_to_go = True
        self.normalize_advantages = True
        self.num_grad_steps_per_target_update = 10
        self.num_target_updates = 10

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
        self.sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sy_ac_na,
                                                                            logits=self.policy_parameters)

        actor_loss = -tf.reduce_sum(self.sy_logprob_n * self.sy_adv_n)
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(actor_loss)

        # define the critic
        self.critic_prediction = tf.squeeze(self.build_mlp(self.sy_ob_no, 1, "nn_critic"))
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)
        self.critic_loss = tf.losses.mean_squared_error(self.sy_target_n, self.critic_prediction)
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

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
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
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
            next_obs.append(ob)
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
                "terminal": np.array(terminals, dtype=np.float32)}
        return path

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        v_n = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: ob_no})
        next_v_n = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: next_ob_no})
        adv_n = re_n + (1 - terminal_n) * self.gamma * next_v_n - v_n
        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n, axis=0)) / np.std(adv_n, axis=0)
        return adv_n

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        n = self.num_grad_steps_per_target_update * self.num_target_updates
        for t in range(0, n):
            if t % self.num_grad_steps_per_target_update == 0:
                next_v_n = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: next_ob_no})
                target_n = re_n + (1 - terminal_n) * self.gamma * next_v_n
            self.sess.run(self.critic_update_op, feed_dict={self.sy_target_n: target_n, self.sy_ob_no: ob_no})

    def update_actor(self, ob_no, ac_na, adv_n):
        self.sess.run(self.actor_update_op,
                      feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_adv_n: adv_n})