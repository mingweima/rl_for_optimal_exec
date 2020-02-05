import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

class DDDQNNet:
    def __init__(self, state_size, action_size, initial_learning_rate, name):
        # define learning rate
        #         self.global_step = tf.Variable(0, trainable=False)
        #         self.initial_learning_rate = initial_learning_rate  # 初始学习率
        #         self.learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=self.global_step, decay_steps=200, decay_rate=0.99, staircase=False)
        #         self.add_global = self.global_step.assign_add(1)

        # define input shapes
        self.learning_rate = initial_learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 100, 120, 4]
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            # define network
            # Input is 100x120x4
            self.conv_first1 = tf.keras.layers.LSTM(64, return_sequences=True, activation='None')(self.inputs_)
            self.conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(self.conv_first1)

            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.keras.layers.LSTM(32, activation='None')(self.conv_first1)
            self.value_fc = tf.keras.layers.LeakyReLU(alpha=0.01)(self.value_fc)
            self.value = tf.keras.layers.Dense(1)(self.value_fc)

            # The one that calculate A(s,a)
            self.advantage_fc = tf.keras.layers.LSTM(32, activation = 'None')(self.conv_first1)
            self.advantage_fc = tf.keras.layers.LeakyReLU(alpha=0.01)(self.advantage_fc)
            self.advantage = tf.keras.layers.Dense(self.action_size)(self.advantage_fc)

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            self.output_softmax = tf.keras.activations.softmax(self.output)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is modified because of PER
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree

            self.loss = tf.reduce_mean(tf.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
                self.loss)
            # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, epsilon=0.1).minimize(self.loss)

            # record my values
            # self.display_action = tf.argmax(self.actions_, axis=1)

    def record_tensorboard(self):
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.histogram("inputs_", self.inputs_)
        tf.summary.histogram("actions_", self.display_action)
        tf.summary.histogram("target_Q", self.target_Q)
        tf.summary.histogram("conv_first1", self.conv_first1)
        tf.summary.histogram("value_fc", self.value_fc)
        tf.summary.histogram("value", self.value)
        tf.summary.histogram("adv_fc", self.advantage_fc)
        tf.summary.histogram("adv", self.advantage)
        tf.summary.scalar("loss", self.loss)
        self.merge_opt = tf.summary.merge_all()

        return self.merge_opt

    def get_graph(self):
        return tf.get_default_graph()
