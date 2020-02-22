import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datetime
import datetime
import time
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

    ticker = hyperparameters['ticker']
    look_back = hyperparameters['lstm_lookback']
    liquidate_volume = hyperparameters['liquidate_volume']
    price_smooth = hyperparameters['price_smooth']

    initial_shares = {
        # 'BARC': 31.42e6 * liquidate_volume,
        # 'HSBA': 22.17e6 * liquidate_volume,
        'ULVR': 2.63e6 * liquidate_volume,
        'RDSa': 10.21e6 * liquidate_volume,
        'RR': 4.78e6 * liquidate_volume
    }

    t = time.strftime('%Y-%m-%d_%H:%M:%I', time.localtime(time.time()))
    dirpath = os.getcwd() + '/recordings/all/loop{}_{}'.format(hyperparameters['total_loop'], t)

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    almgren_chriss_f = open(dirpath + '/almgren_chriss.txt', 'w+')

    train_env = {}
    for ticker in initial_shares.keys():
        train_date = {}
        train_dict = {}
        for month in train_months:
            train_dict[month] = {}
            with open(os.getcwd() + '/trading_environment/data/{}/{}.txt'.format(ticker, month), 'rb') as df_train:
                data = pickle.load(df_train, encoding='iso-8859-1')
            date = pd.to_datetime(data['Date-Time'].dt.strftime('%Y/%m/%d'))
            unique_date = pd.unique(date)
            train_date[month] = unique_date
            for day in unique_date:
                train_dict[month][day] = {}
                for session in ['morning', 'afternoon']:
                    with open(os.getcwd() +
                          '/trading_environment/data/{}/{}_{}_{}.txt'.format(ticker, month, day, session), 'rb') as df:
                        data = pickle.load(df, encoding='iso-8859-1')
                        train_dict[month][day][session] = data
        train_env[ticker] = Simulator(train_dict, train_date, ac_dict, ob_dict, initial_shares[ticker], look_back, price_smooth)
    num_of_training_days = sum(len(v) for _, v in train_date.items())

    test_env = {}
    for ticker in initial_shares.keys():
        test_date = {}
        test_dict = {}
        for month in test_months:
            test_dict[month] = {}
            with open(os.getcwd() + '/trading_environment/data/{}/{}.txt'.format(ticker, month), 'rb') as df_test:
                data = pickle.load(df_test, encoding='iso-8859-1')
            date = pd.to_datetime(data['Date-Time'].dt.strftime('%Y/%m/%d'))
            unique_date = pd.unique(date)
            test_date[month] = unique_date
            for day in unique_date:
                test_dict[month][day] = {}
                for session in ['morning', 'afternoon']:
                    with open(os.getcwd() +
                          '/trading_environment/data/{}/{}_{}_{}.txt'.format(ticker, month, day, session), 'rb') as df:
                        data = pickle.load(df, encoding='iso-8859-1')
                        test_dict[month][day][session] = data
        test_env[ticker] = Simulator(test_dict, test_date, ac_dict, ob_dict, initial_shares[ticker], look_back, price_smooth)
    num_of_test_days = sum(len(v) for _, v in test_date.items())

    for f in [None, almgren_chriss_f]:
        print('Training Set Num of Days: ', num_of_training_days, file=f)
        print('Test Set Num of Days: ', num_of_test_days, file=f)
        print('============================================================', file=f)
        print('Running Almgren Chriss!', file=f)

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

    bar = tqdm(range(num_of_training_days * 2 * len(list(initial_shares.keys()))), leave=False)
    bar.set_description('AC Training Set')
    print('Training Set', file=almgren_chriss_f)
    rewards = []
    for ticker in initial_shares.keys():
        res = []
        ps = []
        acs = []
        dones = []
        for month in train_date.keys():
            for day in train_date[month]:
                for session in ['morning', 'afternoon']:
                    bar.update(1)
                    train_env[ticker].reset(month, day, session)
                    total_reward = 0
                    while True:
                        state, reward, done, info = train_env[ticker].step(-2)
                        total_reward += reward
                        res.append(reward)
                        acs.append(info['size'])
                        ps.append(info['price'])
                        if done:
                            dones.append(len(res) - 1)
                            break
                    rewards.append(total_reward)
                    print(ticker, ', {}, {} Total Reward: '.format(day, session), round(total_reward, 3),
                          file=almgren_chriss_f)
        fig = plt.figure(figsize=(40, 20))
        reward_plot = fig.add_subplot(311)
        reward_plot.plot(res)
        reward_plot.set_title('Reward')
        ac_plot = fig.add_subplot(312)
        color = ['b'] * len(acs)
        for indx in dones:
            color[indx] = 'r'
        ac_plot.bar(range(len(acs)), acs, color=color)
        ac_plot.set_title('Action')
        p_plot = fig.add_subplot(313)
        p_plot.set_title('Price')
        p_plot.plot(ps)
        plt.savefig(dirpath + '/AC_train_{}.png'.format(ticker))
    bar.close()

    for f in [None, almgren_chriss_f]:
        print('Train AC Average: ', round(np.average(rewards), 3), file=f)
        print('============================================================', file=f)

    print('Test Set', file=almgren_chriss_f)
    bar = tqdm(range(num_of_test_days * 2 * len(list(initial_shares.keys()))), leave=False)
    bar.set_description('AC Test Set')
    rewards = []
    for ticker in initial_shares.keys():
        res = []
        ps = []
        acs = []
        dones = []
        for month in test_date.keys():
            for day in test_date[month]:
                for session in ['morning', 'afternoon']:
                    bar.update(1)
                    test_env[ticker].reset(month, day, session)
                    total_reward = 0
                    while True:
                        state, reward, done, info = test_env[ticker].step(-2)
                        total_reward += reward
                        res.append(reward)
                        acs.append(info['size'])
                        ps.append(info['price'])
                        if done:
                            dones.append(len(res) - 1)
                            break
                    rewards.append(total_reward)
                    print(ticker, ', {}, {} Total Reward: '.format(day, session), round(total_reward, 3),
                          file=almgren_chriss_f)

        fig = plt.figure(figsize=(40, 20))
        reward_plot = fig.add_subplot(311)
        reward_plot.plot(res)
        reward_plot.set_title('Reward')
        ac_plot = fig.add_subplot(312)
        color = ['b'] * len(acs)
        for indx in dones:
            color[indx] = 'r'
        ac_plot.bar(range(len(acs)), acs, color=color)
        ac_plot.set_title('Action')
        p_plot = fig.add_subplot(313)
        p_plot.set_title('Price')
        p_plot.plot(ps)
        plt.savefig(dirpath + '/AC_test_{}.png'.format(ticker))
    bar.close()

    for f in [None, almgren_chriss_f]:
        print('Test AC Average: ', round(np.average(rewards), 3), file=f)
        print('============================================================', file=f)

    AC_list_f = open(dirpath + '/AC_list_f.txt', 'wb')
    pickle.dump(rewards, AC_list_f)
    AC_list_f.close()

    for f in [None, almgren_chriss_f]:
        print('============================================================', file=f)
        print('Running Hothead!', file=f)
    bar = tqdm(range(num_of_test_days * 2 * len(list(initial_shares.keys()))), leave=False)
    bar.set_description('Hothead Test Set')
    # Hothead
    rewards = []
    for ticker in initial_shares.keys():
        res = []
        ps = []
        acs = []
        for month in test_date.keys():
            for day in test_date[month]:
                for session in ['morning', 'afternoon']:
                    bar.update(1)
                    test_env[ticker].reset(month, day, session)
                    total_reward = 0
                    state, reward, done, info = test_env[ticker].step(-1)
                    total_reward += reward
                    res.append(reward)
                    acs.append(info['size'])
                    ps.append(info['price'])
                    rewards.append(total_reward)
                    print(ticker, ', {}, {} Total Reward: '.format(day, session), round(total_reward, 3), file=almgren_chriss_f)
                    # print(ticker, ', {}, {} Total Reward: '.format(day, session), round(total_reward, 3))
        fig = plt.figure(figsize=(40, 20))
        reward_plot = fig.add_subplot(311)
        reward_plot.plot(res)
        reward_plot.set_title('Reward')
        ac_plot = fig.add_subplot(312)
        ac_plot.bar(range(len(acs)), acs)
        ac_plot.set_title('Action')
        p_plot = fig.add_subplot(313)
        p_plot.set_title('Price')
        p_plot.plot(ps)
        plt.savefig(dirpath + '/Hothead_test_{}.png'.format(ticker))
    bar.close()

    for f in [None, almgren_chriss_f]:
        print('Hothead Average: ', round(np.average(rewards), 3), file=f)
        print('============================================================', file=f)

    def te_performance(month, day, session, ticker):
        state = test_env[ticker].reset(month, day, session)
        state = np.array(state)
        all_reward = []
        while True:
            Qs = sess.run(DQNetwork.output_softmax, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]
            next_state, reward, done, info = test_env[ticker].step(np.argmax(action))
            all_reward.append(reward)
            if done:
                break
            else:
                # If not done, the next_state become the current state
                next_state = np.array(next_state)
                state = next_state
        return np.sum(all_reward)

    almgren_chriss_f.close()

    print('Training Network!')

    state = train_env[list(initial_shares.keys())[0]].reset(list(train_date.keys())[0], train_date[list(train_date.keys())[0]][0], 'morning')
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
    target_network_update = hyperparameters['target_network_update']
    network_update = hyperparameters['network_update']

    ### MEMORY HYPERPARAMETERS
    memory_size = hyperparameters['memory_size']  # Number of experiences the Memory can keep

    config_f = open(dirpath + '/config.txt', 'w+')
    print('Hyperparameters', file=config_f)
    print('Ticker: ', initial_shares.keys(), file=config_f)
    print('Memory Size: {}'.format(memory_size), file=config_f)
    print('Total Shares to Liquidate: {}'.format(liquidate_volume), file=config_f)
    print('LSTM Lookback: {}'.format(look_back), file=config_f)
    print('Total Loop: {}'.format(total_loop), file=config_f)
    print('Batch Size: {}'.format(batch_size), file=config_f)
    print('Final Exploration Probability: {}'.format(explore_stop), file=config_f)
    print('Initial Learning Rate: {}'.format(initial_learning_rate), file=config_f)
    print('Exploration Decay Rate: {}'.format(decay_rate), file=config_f)
    print('Update Target Network Period (step): {}'.format(target_network_update), file=config_f)
    print('Update Network Period (step): {}'.format(network_update), file=config_f)
    print('Observation Space: ', ob_dict, file=config_f)
    print('Number of Train Months: {}'.format(hyperparameters['num_of_train_months']), file=config_f)
    print('Number of Test Months: {}'.format(hyperparameters['num_of_test_months']), file=config_f)
    print('Observation Space: \n', ob_dict, file=config_f)
    print('Action Space: \n', ac_dict, file=config_f)
    config_f.close()

    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True

    # config = tf.ConfigProto(
    #     device_count={"CPU": 20},
    #     inter_op_parallelism_threads=20,
    #     intra_op_parallelism_threads=20)
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


    while len(memory.buffer) < batch_size:
        bar = tqdm(range(num_of_training_days * 2), leave=False)
        bar.set_description('Pretraining')
        for month in train_date.keys():
            for day in train_date[month]:
                for session in ['morning', 'afternoon']:
                    for ticker in initial_shares.keys():
                        state = np.array(train_env[ticker].reset(month, day, session))
                        while True:
                            choice = random.randint(1, len(possible_actions)) - 1
                            action = possible_actions[choice]
                            next_state, reward, done, _ = train_env[ticker].step(np.argmax(action))
                            memory.add((state, action, reward, next_state, done))
                            state = next_state
                            if done:
                                break
                    bar.update(1)
        bar.close()

    # writer = tf.summary.FileWriter('./tensorboard', DQNetwork.get_graph())

    ## Losses

    # write_op = DQNetwork.record_tensorboard()

    def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        # explore_probability = explore_start * np.exp( - decay_rate * decay_step)
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

    saver = tf.train.Saver()

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

    train_f = open(dirpath + '/training.txt', 'a+')
    test_f = open(dirpath + '/test.txt', 'a+')
    print('Training Network!\n', file=train_f)
    print('Training Network!\n', file=test_f)
    train_f.close()
    test_f.close()

    for loop in range(1, total_loop + 1):
        loop_indx += 1
        decay_step += 1
        total_reward_list = []
        losses = []

        months = list(train_date.keys())
        np.random.shuffle(months)
        # months = np.random.choice(months, target_network_update)

        bar = tqdm(range(num_of_training_days * 2), leave=False)
        bar.set_description('Running Loop {}'.format(loop))

        num_of_day = 0
        for month in months:
            days = train_date[month]
            np.random.shuffle(days)
            for day in days:
                num_of_day += 1
                for session in ['morning', 'afternoon']:
                    bar.update(1)
                    for ticker in initial_shares.keys():
                        step = 0
                        episode_rewards = []
                        state = train_env[ticker].reset(month, day, session)
                        state = np.array(state)
                        while step < max_steps:
                            total_step += 1
                            step += 1
                            # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                            try:
                                action, explore_probability = predict_action(explore_start,
                                                                    explore_stop,
                                                                    decay_rate,
                                                                    decay_step,
                                                                    state,
                                                                    possible_actions)
                            except:
                                raise Exception('Model Output Nan Probability!')
                            # Do the action
                            next_state, reward, done, _ = train_env[ticker].step(np.argmax(action))
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
                            q_target_next_state = sess.run(TargetNetwork.output,
                                                       feed_dict={TargetNetwork.inputs_: next_states_mb})
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

                            if total_step % target_network_update == 0:
                                update_target = update_target_graph()
                                sess.run(update_target)
                # Write TF Summaries
                #     summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                #                                         DQNetwork.target_Q: targets_mb,
                #                                         DQNetwork.actions_: actions_mb})
                #
                #     writer.add_summary(summary, total_step)
                #     total_step += 1

        bar.close()

        print(f'{datetime.datetime.now()} '
              f'Loop = {loop}, '
              f'Avg R = {round(np.average(total_reward_list), 3)}, '
              f'Loss = {round(np.average(losses), 3)}, '
              f'Explore P = {explore_probability}')

        bar = tqdm(range(num_of_test_days * 2), leave=False)
        bar.set_description("Testing Results")
        reward_list = []
        for month in test_date.keys():
            for day in test_date[month]:
                for session in ['morning', 'afternoon']:
                    bar.update(1)
                    for ticker in initial_shares.keys():
                        check_reward = te_performance(month, day, session, ticker)
                        reward_list.append(check_reward)
        bar.close()
        avg_re = np.average(reward_list)
        test_f = open(dirpath + '/test.txt', 'a+')
        print('Loop {}, Test Average Reward: '.format(loop_indx), round(avg_re, 3), '\n', file=test_f)
        test_f.close()
        print('Loop {}, Test Average Reward: '.format(loop_indx), round(avg_re, 3))
        test_avg_reward.append(avg_re)
        saver.save(sess, dirpath + '/loop{}_model.ckpt'.format(loop_indx))

        avg_re_per_loop.append(np.mean(total_reward_list))
        loss_per_loop.append(np.average(losses))

        train_f = open(dirpath + '/training.txt', 'a+')
        print(f'{datetime.datetime.now()} '
              f'Loop = {loop}, '
              f'Avg R = {round(np.average(total_reward_list), 3)}, '
              f'Loss = {round(np.average(losses), 3)}, '
              f'Explore P = {explore_probability}\n', file=train_f)
        train_f.close()

        if loop_indx % 10 == 0:
            fig1 = plt.figure()
            reward_plot = fig1.add_subplot(111)
            reward_plot.plot(avg_re_per_loop)
            reward_plot.set_title('Blue: Training Set Reward; Red: Test Set Reward')
            test_plot = reward_plot.twinx()
            test_plot.plot(test_avg_reward, color='r', linestyle='dashed')
            plt.savefig(dirpath + '/reward_{}.png'.format(loop_indx))

            fig2 = plt.figure()
            loss_plot = fig2.add_subplot(111)
            loss_plot.plot(loss_per_loop)
            loss_plot.set_title('Loss')
            plt.savefig(dirpath + '/loss_{}.png'.format(loop_indx))

            for ticker in initial_shares.keys():
                res = []
                ps = []
                acs = []
                dones = []
                file = open(dirpath + '/loop{}_{}.txt'.format(loop_indx, ticker), 'a+')
                for month in test_date.keys():
                    for day in test_date[month]:
                        for session in ['morning', 'afternoon']:
                            state = test_env[ticker].reset(month, day, session)
                            state = np.array(state)
                            step = 1
                            while True:
                                Qs = sess.run(DQNetwork.output_softmax,
                                              feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
                                choice = np.argmax(Qs)
                                action = possible_actions[int(choice)]
                                next_state, reward, done, info = test_env[ticker].step(np.argmax(action))
                                res.append(reward)
                                acs.append(info['size'])
                                ps.append(info['price'])
                                print(ticker, ', day: {}, {}, step: {}, reward: {}, size: {}, price: {} '.format(
                                    day, session, step, reward, info['size'], info['price']), file=file)
                                if done:
                                    dones.append(len(res) - 1)
                                    break
                                else:
                                    next_state = np.array(next_state)
                                    state = next_state
                                    step += 1
                file.close()
                fig = plt.figure(figsize=(40, 20))
                reward_plot = fig.add_subplot(311)
                reward_plot.plot(res)
                reward_plot.set_title('Reward')
                ac_plot = fig.add_subplot(312)
                color = ['b'] * len(acs)
                for indx in dones:
                    color[indx] = 'r'
                ac_plot.bar(range(len(acs)), acs, color=color)
                ac_plot.set_title('Action')
                p_plot = fig.add_subplot(313)
                p_plot.set_title('Price')
                p_plot.plot(ps)
                plt.savefig(dirpath + '/loop{}_{}.png'.format(loop_indx, ticker))

    fig1 = plt.figure()
    reward_plot = fig1.add_subplot(111)
    reward_plot.plot(avg_re_per_loop)
    reward_plot.set_title('Blue: Training Set Reward; Red: Test Set Reward')
    test_plot = reward_plot.twinx()
    test_plot.plot(test_avg_reward, color='r', linestyle='dashed')
    plt.savefig(dirpath + '/reward.png')

    fig2 = plt.figure()
    loss_plot = fig2.add_subplot(111)
    loss_plot.plot(loss_per_loop)
    loss_plot.set_title('Loss')
    plt.savefig(dirpath + '/loss.png')

    print('============================================================')
    test_f = open(dirpath + '/test.txt', 'a+')
    print('============================================================', file=test_f)
    reward_list = []
    for month in test_date.keys():
        for day in test_date[month]:
            for session in ['morning', 'afternoon']:
                for ticker in initial_shares.keys():
                    check_reward = te_performance(month, day, session, ticker)
                    print(ticker, ', {} Total Reward: '.format(day), check_reward, file=test_f)
                    print(ticker, ', {} Total Reward: '.format(day), check_reward)
                    reward_list.append(check_reward)

    test_list_f = open(dirpath + '/test_list_f.txt', 'wb')
    pickle.dump(reward_list, test_list_f)
    test_list_f.close()

    print('Test Average Reward: ', round(np.average(reward_list), 3))
    print('Test Average Reward: ', round(np.average(reward_list), 3), file=test_f)

    saver.save(sess, dirpath + '/model.ckpt')
    sess.close()
    test_f.close()

