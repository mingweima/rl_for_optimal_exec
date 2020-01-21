# -*- coding: utf-8 -*-

import tensorflow as tf             # Deep Learning library
import numpy as np                  # Handle matrices
from collections import deque       # Ordered collection with ends
import random
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import OneHotEncoder


from simple_test.simple_env import Simulator


train_data_path = '/nfs/home/mingweim/lob/hsbc/L2_HSBA.L_2018-09-01_2018-09-30.csv.gz'
test_data_path = '/nfs/home/mingweim/lob/hsbc/L2_HSBA.L_2018-10-01_2018-10-31.csv.gz'
train_raw_data = pd.read_csv(train_data_path, compression='gzip', error_bad_lines=False)
test_raw_data = pd.read_csv(test_data_path, compression='gzip', error_bad_lines=False)

data_list = []

for raw_data in [train_raw_data, test_raw_data]:
    data = raw_data.drop(['#RIC', 'Domain', 'GMT Offset', 'Type', 'L1-BuyNo', 'L1-SellNo', 'L2-BuyNo', 'L2-SellNo',
                          'L3-BuyNo', 'L3-SellNo', 'L4-BuyNo', 'L4-SellNo', 'L5-BuyNo', 'L5-SellNo',
                          'L6-BuyNo', 'L6-SellNo', 'L7-BuyNo', 'L7-SellNo', 'L8-BuyNo', 'L8-SellNo',
                          'L9-BuyNo', 'L9-SellNo', 'L10-BuyNo', 'L10-SellNo'], axis=1)
    data['Date-Time'] = pd.to_datetime(data['Date-Time'],
                                       format='%Y-%m-%dT%H:%M:%S.%fZ').dt.round('{}s'.format(600))
    data = data.groupby(['Date-Time']).first().reset_index()
    data['Day'] = data['Date-Time'].dt.dayofweek
    data = data.drop(data.loc[(data['Day'] == 5) | (data['Day'] == 6)].index)
    data['Hour'] = data['Date-Time'].dt.hour
    data['Minute'] = data['Date-Time'].dt.minute
    data = data.drop(
        data.loc[(data['Hour'] < 8) | (data['Hour'] > 16) | ((data['Hour'] == 16) & (data['Minute'] > 0))].index)
    data = data.drop(['Hour', 'Minute', 'Day'], axis=1)
    data = data.iloc[1:]
    data_list.append(data)
    date = pd.to_datetime(data['Date-Time'].dt.strftime('%Y/%m/%d'))
    unique_date = pd.unique(date)
    print(unique_date)

train_data = data_list[0]
test_data = data_list[1]

def Almgren_Chriss(kappa, ac_dict, step, num_of_steps):
        def closest_action(nj):
            action = 0
            difference = abs(ac_dict[action] - nj)
            for ac, proportion in ac_dict.items():
                if type(ac) is int:
                    if (proportion - nj) < difference:
                        action = ac
                        difference = abs(ac_dict[action] - nj)
            return action

        if step == num_of_steps:
            nj = 1
        elif kappa == 0:
            nj = 1 / num_of_steps
        else:
            nj = 2 * np.sinh(0.5 * kappa) * np.cosh(kappa * (
                    num_of_steps - (step - 0.5))) / np.sinh(kappa * num_of_steps)
        action = closest_action(nj)
        return action

print('========================================')
print('Running Almgren Chriss!')
print('========================================')

# ac_dict = {0: 0, 1: 0.005, 2: 0.01, 3: 0.015, 4: 0.02, 5: 0.025,
#            6: 0.03, 7: 0.035, 8: 0.04, 9: 0.05, 10: 0.06, 11: 0.07,
#            12: 0.08, 13: 0.09, 14: 0.10, 15: 0.12, 16: 0.14, 17: 0.2, 18: 0.3, 19: 0.5, 20: 1}

# ac_dict = {0: 0, 1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.1,
#            6: 0.15, 7: 0.2, 8: 0.25, 9: 0.5, 10: 1}

ac_dict = {0: 0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1, 5: 1.25,
           6: 1.5, 7: 1.75, 8: 2}

# ac_dict = {0: 0, 1: 0.2, 2: 0.5, 3: 1, 4: 1.5, 5: 2, 6: 2.5, 7: 3, 8: 3.5, 9: 4, 10: 6, 11: 8, 12: 10}

# ac_dict = {0: 1}

### TRAINING HYPERPARAMETERS
total_loop = 1
total_episodes = 20
max_steps = 100000              # Max possible steps in an episode
batch_size = 128                # Batch size


env_train = Simulator(train_data, ac_dict)
rewards = []
for num_days in range(total_episodes):
    env_train.reset(num_days=num_days)
    total_reward = 0
    for step in np.arange(1, 31):
        action = Almgren_Chriss(0, ac_dict, step, 30)
        state, reward, done, _ = env_train.step(4)
        total_reward += reward
    rewards.append(total_reward)
    print('AC day {}:'.format(num_days + 1), total_reward)
print('AC Average: ', np.average(rewards))
print('========================================')

env_test = Simulator(test_data, ac_dict)
rewards = []
for num_days in range(20):
    env_test.reset(num_days=num_days)
    total_reward = 0
    for step in np.arange(1, 31):
        action = Almgren_Chriss(0, ac_dict, step, 30)
        state, reward, done, _ = env_test.step(4)
        total_reward += reward
    rewards.append(total_reward)
    print('AC day {}:'.format(num_days + 1), total_reward)
print('AC Average: ', np.average(rewards))
print('========================================')

def te_performance(which_day):

    state = env_test.reset(which_day)
    state = np.array(state)
    all_reward = []

    while True:
        Qs = sess.run(DQNetwork.output_softmax, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
        next_state, reward, done, _ = env_test.step(np.argmax(action))
        all_reward.append(reward)

        if done:
            break
        else:
            # If not done, the next_state become the current state
            next_state = np.array(next_state)
            state = next_state

    return np.sum(all_reward)

print('Training Network!')

env_train = Simulator(train_data, ac_dict)
state = env_train.reset(num_days=0)
state = np.array(state)
#state = state.reshape(state.shape + (1,))

action_size = len(ac_dict)
enc = OneHotEncoder()
possible_actions = enc.fit_transform([[i] for i in range(action_size)])
possible_actions = possible_actions.toarray()
# possible_actions = [[1,0], [0,1]]

### MODEL HYPERPARAMETERS
state_size = state.shape      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = len(possible_actions)  # 8 possible actions
initial_learning_rate = 0.005    # Alpha (aka learning rate)

# Exploration parameters for epsilon greedy strategy
initial_exploration_steps = 0
explore_start = 1.0            # exploration probability at start
explore_stop = 0           # minimum exploration probability
decay_rate = 0.0005           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99                    # Discounting rate
loop_update = 3

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 100000          # Number of experiences the Memory can keep

print('Learning Rate: ', initial_learning_rate)
print('Gamma: ', gamma)
print('Memory Size: ', memory_size)
print('Batch Size: ', batch_size)
print('Explore Stop:', explore_stop)
print('Explore Decay', decay_rate)

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


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
            self.conv_first1 = tf.keras.layers.LSTM(64, return_sequences=True)(self.inputs_)
            self.conv_first1 = tf.keras.layers.LeakyReLU(alpha=0.01)(self.conv_first1)
            self.conv_first1 = tf.keras.layers.Dropout(0.2)(self.conv_first1)
            self.conv_first1 = tf.keras.layers.Dense(16)(self.conv_first1)
            self.conv_first1 = tf.keras.layers.Dropout(0.2)(self.conv_first1)

            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.keras.layers.LSTM(32)(self.conv_first1)
            self.value_fc = tf.keras.layers.Dense(8)(self.value_fc)
            self.value_fc = tf.keras.layers.LeakyReLU(alpha=0.01)(self.value_fc)
            self.value_fc = tf.keras.layers.Dropout(0.2)(self.value_fc)
            self.value = tf.keras.layers.Dense(1)(self.value_fc)

            #             The one that calculate A(s,a)
            self.advantage_fc = tf.keras.layers.LSTM(32)(self.conv_first1)
            self.advantage_fc = tf.keras.layers.Dense(16)(self.advantage_fc)
            self.advantage_fc = tf.keras.layers.LeakyReLU(alpha=0.01)(self.advantage_fc)
            self.advantage_fc = tf.keras.layers.Dropout(0.2)(self.advantage_fc)
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
            self.display_action = tf.argmax(self.actions_, axis=1)

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

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DDDQNNet(state_size, action_size, initial_learning_rate, name="DQNetwork")

# Instantiate the target network
TargetNetwork = DDDQNNet(state_size, action_size, initial_learning_rate, name="TargetNetwork")

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]

memory = Memory(max_size=memory_size)

num_of_steps = 0
while num_of_steps < pretrain_length:
    for episode in np.arange(total_episodes):
        state = np.array(env_train.reset(episode))
        while True:
            choice = random.randint(1, len(possible_actions)) - 1
            action = possible_actions[choice]
            next_state, reward, done, _ = env_train.step(np.argmax(action))
            memory.add((state, action, reward, next_state, done))
            state = next_state
            num_of_steps += 1
            if done:
                break

writer = tf.summary.FileWriter('./tensorboard', DQNetwork.get_graph())

## Losses

write_op = DQNetwork.record_tensorboard()


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    # if explore_probability > exp_exp_tradeoff:
    #     # Make a random action (exploration)
    #     choice = random.randint(1, len(possible_actions)) - 1
    #     action = possible_actions[choice]
    if explore_probability > exp_exp_tradeoff:
        Ps = sess.run(DQNetwork.output_softmax, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})[0]
        choice = np.random.choice(action_size, 1, p=Ps)[0]
        action = possible_actions[choice]
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability

def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder


saver = tf.train.Saver(max_to_keep=total_loop)

total_step = 0
decay_step = 0
loop_indx = 0

rewards_list = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())
update_target = update_target_graph()
sess.run(update_target)

avg_re_per_loop = []
loss_per_loop = []
test_avg_reward = []

avg_re_matrix = [[] for _ in range(20)]

for loop in range(total_loop):
    loop_indx += 1
    total_reward_list = []
    losses = []

    total_epi = np.arange(total_episodes)
    np.random.shuffle(total_epi)

    total_reward_dict = {}


    for episode in total_epi:

        # Set step to 0
        step = 0

        # Initialize the rewards of the episode
        episode_rewards = []

        # Make a new episode and observe the first state
        state = env_train.reset(num_days=episode)
        state = np.array(state)
#         state = state.reshape(state.shape + (1,))
 
        while step < max_steps:
            step += 1

            # Increase decay_step
            decay_step += 1

            # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
            action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                         possible_actions)

            # Do the action
            next_state, reward, done, _ = env_train.step(np.argmax(action))
            next_state = np.array(next_state)
#             next_state = next_state.reshape(next_state.shape + (1,))

            episode_rewards.append(reward)

            # If the game is finished
            if done:
                # Set step = max_steps to end the episode
                step = max_steps
                memory.add((state, action, reward, next_state, done))
                total_reward = np.sum(episode_rewards)
                total_reward_list.append(total_reward)
                
#                 print(f'Loop = {loop},'
#                         f'Episode = {episode},'
#                         f'total reward = {total_reward},'
#                         f'Training loss = {loss},'
#                         f'Explore P = {explore_probability}')

                rewards_list.append((loop, episode, total_reward))

            else:
                memory.add((state, action, reward, next_state, done))
                state = next_state

            ### LEARNING PART
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            ### DOUBLE DQN Logic
            # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
            # Use TargetNetwork to calculate the Q_val of Q(s',a')

            # Get Q values for next_state
            q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

            # Calculate Qtarget for all actions that state
            q_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})

            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                # We got a'
                action = np.argmax(q_next_state[i])

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])

                else:
                    # Take the Qtarget for action a'
                    target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                    target_Qs_batch.append(target)

            targets_mb = np.array([each for each in target_Qs_batch])

            _, loss = sess.run([DQNetwork.optimizer, DQNetwork.loss],
                               feed_dict={DQNetwork.inputs_: states_mb,
                                          DQNetwork.target_Q: targets_mb,
                                          DQNetwork.actions_: actions_mb})

            losses.append(loss)

            # Write TF Summaries
            summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                    DQNetwork.target_Q: targets_mb,
                                                    DQNetwork.actions_: actions_mb})

            writer.add_summary(summary, total_step)
            total_step += 1

    if loop_indx % loop_update == 0:
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)
        tau = 0
        print("Model updated")

        reward_list = []
        for day in range(20):
            check_reward = te_performance(which_day=day)
            reward_list.append(check_reward)
            avg_re_matrix[day].append(check_reward)


        avg_re = np.average(reward_list)
        print('Test Average Reward: ', avg_re)
        test_avg_reward.append(avg_re)
        avg_re_per_loop.append(np.mean(total_reward_list))
        loss_per_loop.append(np.average(losses))

    print(f'Loop = {loop}, '
        f'Avg R = {np.mean(total_reward_list)}, '
        f'Max R = {np.max(total_reward_list)}, '
        f'Min R = {np.min(total_reward_list)}, '
        f'Loss = {np.average(losses)}, '
        f'Explore P = {explore_probability}')


fig = plt.figure()
reward_plot = fig.add_subplot(121)
reward_plot.plot(avg_re_per_loop)
reward_plot.set_title('Reward')
test_plot = reward_plot.twinx()
test_plot.plot(test_avg_reward, color='r', linestyle='dashed')

# for i in range(20):
#     mat = reward_plot.twinx()
#     mat.plot(avg_re_matrix[i])

loss_plot = fig.add_subplot(122)
loss_plot.plot(loss_per_loop)
loss_plot.set_title('Loss')
plt.savefig('plot1.png')
plt.show()

print('========================================')
reward_list = []
for day in range(22):
    check_reward = te_performance(which_day=day)
    print('Day {}'.format(day+1), check_reward)
    reward_list.append(check_reward)
print('Train Average Reward: ', np.average(reward_list))
print('========================================')

reward_list = []
for day in range(20):
    check_reward = te_performance(which_day=day)
    print('Test Day {}'.format(day+1), check_reward)
    reward_list.append(check_reward)
print('Test Average Reward: ', np.average(reward_list))
