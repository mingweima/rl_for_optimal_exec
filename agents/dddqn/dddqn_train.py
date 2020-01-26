import datetime
import pickle
import random

from tqdm import tqdm
import tensorflow as tf
import numpy as np
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import OneHotEncoder

from trading_environment.trading_env import Simulator
from agents.dddqn.dddqnnet import DDDQNNet
from agents.dddqn.memory import Memory

def dddqn_train(hyperparameters, ac_dict, ob_dict, train_months, test_months):

    train_dict = {}
    train_date = {}
    for month in train_months:
        train_dict[month] = {}
        with open('/nfs/home/mingweim/rl_for_optimal_exec'
                  '/trading_environment/data/HSBA/{}.txt'.format(month), 'rb') as df_train:
            data = pickle.load(df_train, encoding='iso-8859-1')
        date = pd.to_datetime(data['Date-Time'].dt.strftime('%Y/%m/%d'))
        unique_date = pd.unique(date)
        train_date[month] = unique_date
        for day in unique_date:
            with open('/nfs/home/mingweim/rl_for_optimal_exec'
                      '/trading_environment/data/HSBA/{}_{}.txt'.format(month, day), 'rb') as df:
                data = pickle.load(df, encoding='iso-8859-1')
            train_dict[month][day] = data
    num_of_training_days = sum(len(v) for _, v in train_date.items())

    print('========================================')
    print('Training Set Num of Days: ', num_of_training_days)
    print('========================================')

    test_dict = {}
    test_date = {}
    for month in test_months:
        test_dict[month] = {}
        with open('/nfs/home/mingweim/rl_for_optimal_exec'
                  '/trading_environment/data/HSBA/{}.txt'.format(month), 'rb') as df_test:
            data = pickle.load(df_test, encoding='iso-8859-1')
        date = pd.to_datetime(data['Date-Time'].dt.strftime('%Y/%m/%d'))
        unique_date = pd.unique(date)
        test_date[month] = unique_date
        for day in unique_date:
            with open('/nfs/home/mingweim/rl_for_optimal_exec'
                      '/trading_environment/data/HSBA/{}_{}.txt'.format(month, day), 'rb') as df:
                data = pickle.load(df, encoding='iso-8859-1')
            test_dict[month][day] = data
    num_of_test_days = sum(len(v) for _, v in test_date.items())

    print('Test Set Num of Days: ', num_of_test_days)
    print('========================================')
    print('Running Almgren Chriss!')
    print('========================================')

    def almgren_chriss(kappa, ac_dict, step, num_of_steps):
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


    ### TRAINING HYPERPARAMETERS
    total_loop = hyperparameters['total_loop']
    total_episodes = num_of_training_days
    max_steps = 100000  # Max possible steps in an episode
    batch_size = hyperparameters['batch_size']

    print('Training Set')
    env_train = Simulator(train_dict, train_date, ac_dict, ob_dict)
    rewards = []
    for month in train_date.keys():
        for day in train_date[month]:
            env_train.reset(month, day)
            total_reward = 0
            for step in np.arange(1, 31):
                action = almgren_chriss(0, ac_dict, step, 30)
                state, reward, done, _ = env_train.step(4)
                total_reward += reward
            rewards.append(total_reward)
            print('{} Total Reward: '.format(day), total_reward)
    print('AC Average: ', np.average(rewards))
    print('========================================')

    print('Test Set')
    env_test = Simulator(test_dict, test_date, ac_dict, ob_dict)
    rewards = []
    for month in test_date.keys():
        for day in test_date[month]:
            env_test.reset(month, day)
            total_reward = 0
            for step in np.arange(1, 31):
                action = almgren_chriss(0, ac_dict, step, 30)
                state, reward, done, _ = env_test.step(4)
                total_reward += reward
            rewards.append(total_reward)
            print('{} Total Reward: '.format(day), total_reward)
    print('AC Average: ', np.average(rewards))
    print('========================================')

    def te_performance(month, day):
        state = env_test.reset(month, day)
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

    state = env_train.reset(list(train_date.keys())[0], train_date[list(train_date.keys())[0]][0])
    state = np.array(state)
    # state = state.reshape(state.shape + (1,))

    action_size = len(ac_dict)
    enc = OneHotEncoder()
    possible_actions = enc.fit_transform([[i] for i in range(action_size)])
    possible_actions = possible_actions.toarray()
    # possible_actions = [[1,0], [0,1]]

    ### MODEL HYPERPARAMETERS
    state_size = state.shape  # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
    action_size = len(possible_actions)  # 8 possible actions
    initial_learning_rate = hyperparameters['learning_rate']  # Alpha (aka learning rate)

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0  # exploration probability at start
    explore_stop = hyperparameters['explore_stop']  # minimum exploration probability
    decay_rate = hyperparameters['decay_rate']  # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.99  # Discounting rate
    loop_update = hyperparameters['loop_update']

    ### MEMORY HYPERPARAMETERS
    memory_size = hyperparameters['memory_size']  # Number of experiences the Memory can keep

    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Reset the graph
    tf.reset_default_graph()

    # Instantiate the DQNetwork
    DQNetwork = DDDQNNet(state_size, action_size, initial_learning_rate, name="DQNetwork")

    # Instantiate the target network
    TargetNetwork = DDDQNNet(state_size, action_size, initial_learning_rate, name="TargetNetwork")

    memory = Memory(max_size=memory_size)

    bar = tqdm(range(num_of_training_days), leave=False)
    bar.set_description('Pretraining')

    for month in train_date.keys():
        for day in train_date[month]:
            state = np.array(env_train.reset(month, day))
            while True:
                choice = random.randint(1, len(possible_actions)) - 1
                action = possible_actions[choice]
                next_state, reward, done, _ = env_train.step(np.argmax(action))
                memory.add((state, action, reward, next_state, done))
                state = next_state
                if done:
                    bar.update(1)
                    break
    bar.close()

    writer = tf.summary.FileWriter('./tensorboard', DQNetwork.get_graph())

    ## Losses

    write_op = DQNetwork.record_tensorboard()

    def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_start - decay_rate * decay_step
        if explore_probability < explore_stop:
            explore_probability = explore_stop

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

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    update_target = update_target_graph()
    sess.run(update_target)

    avg_re_per_loop = []
    loss_per_loop = []
    test_avg_reward = []

    for loop in range(1, total_loop + 1):
        loop_indx += 1
        decay_step += 1
        total_reward_list = []
        losses = []
        total_epi = np.arange(total_episodes)
        np.random.shuffle(total_epi)

        months = list(train_date.keys())
        np.random.shuffle(months)

        bar = tqdm(range(num_of_training_days), leave=False)
        bar.set_description('Running Loop {}'.format(loop))

        for month in months:

            days = train_date[month]
            np.random.shuffle(days)
            for day in days:
                bar.update(1)
                step = 0
                episode_rewards = []
                state = env_train.reset(month, day)
                state = np.array(state)
                while step < max_steps:
                    step += 1
                    # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                    action, explore_probability = predict_action(explore_start,
                                                                 explore_stop,
                                                                 decay_rate,
                                                                 decay_step,
                                                                 state,
                                                                 possible_actions)
                    # Do the action
                    next_state, reward, done, _ = env_train.step(np.argmax(action))
                    next_state = np.array(next_state)
                    episode_rewards.append(reward)
                    # If the game is finished
                    if done:
                        # Set step = max_steps to end the episode
                        step = max_steps
                        memory.add((state, action, reward, next_state, done))
                        total_reward = np.sum(episode_rewards)
                        total_reward_list.append(total_reward)

                    else:
                        memory.add((state, action, reward, next_state, done))
                        state = next_state

                ### Training Network
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

        bar.close()

        print(f'{datetime.datetime.now()} '
              f'Loop = {loop}, '
              f'Avg R = {np.mean(total_reward_list)}, '
              f'Max R = {np.max(total_reward_list)}, '
              f'Min R = {np.min(total_reward_list)}, '
              f'Loss = {np.average(losses)}, '
              f'Explore P = {explore_probability}')

        if loop_indx % loop_update == 0:
            # Update the parameters of our TargetNetwork with DQN_weights
            update_target = update_target_graph()
            sess.run(update_target)
            tau = 0
            print(f"Model updated at time {datetime.datetime.now()}")

            bar = tqdm(range(num_of_test_days), leave=False)
            bar.set_description("Testing Results")
            reward_list = []
            for month in test_date.keys():
                for day in test_date[month]:
                    bar.update(1)
                    check_reward = te_performance(month, day)
                    reward_list.append(check_reward)
            bar.close()

            avg_re = np.average(reward_list)
            print('Test Average Reward: ', avg_re)
            test_avg_reward.append(avg_re)
            avg_re_per_loop.append(np.mean(total_reward_list))
            loss_per_loop.append(np.average(losses))

    fig = plt.figure()
    reward_plot = fig.add_subplot(121)
    reward_plot.plot(avg_re_per_loop)
    reward_plot.set_title('Reward')
    test_plot = reward_plot.twinx()
    test_plot.plot(test_avg_reward, color='r', linestyle='dashed')

    loss_plot = fig.add_subplot(122)
    loss_plot.plot(loss_per_loop)
    loss_plot.set_title('Loss')
    plt.savefig('plot_ddqn_loop200_bs2000_mem200000.png')
    plt.show()

    print('========================================')
    reward_list = []
    for month in test_date.keys():
        for day in test_date[month]:
            check_reward = te_performance(month, day)
            print('{} Total Reward: '.format(day), check_reward)
            reward_list.append(check_reward)
    print('Test Average Reward: ', np.average(reward_list))

