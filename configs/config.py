import argparse

import matplotlib.pyplot as plt

from agents.dqn.dqn_train import DQNTrain
from agents.drqn.drqn_train import DRQNTrain
from agents.dpg.dpg_train import DPG_Train
from agents.a2c.a2c_train import A2CTrain
from agents.almgren_chriss.almgren_chriss_train import AlmgrenChrissTrain
from agents.dddqn.dddqn_train import dddqn_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='dddqn')
    parser.add_argument('--total_loop', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--explore_stop', type=float, default=0.05)
    parser.add_argument('--decay_rate', type=float, default=0.05)
    parser.add_argument('--target_network_update', type=int, default=500)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--network_update', type=int, default=1)
    parser.add_argument('--ticker', type=str, default='HSBA')
    parser.add_argument('--lstm_lookback', type=int, default=24)
    parser.add_argument('--liquidate_volume', type=float, default=0.05)
    parser.add_argument('--num_of_train_months', type=int, default=5)
    parser.add_argument('--num_of_test_months', type=int, default=2)
    parser.add_argument('--price_smooth', type=int, default=10)

    args = parser.parse_args()
    hyperparameters = {
        'ticker': args.ticker,
        'total_loop': args.total_loop,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'explore_stop': args.explore_stop,
        'decay_rate': args.decay_rate,
        'target_network_update': args.target_network_update,
        'memory_size': args.memory_size,
        'network_update': args.network_update,
        'lstm_lookback': args.lstm_lookback,
        'liquidate_volume': args.liquidate_volume,
        'num_of_train_months': args.num_of_train_months,
        'num_of_test_months': args.num_of_test_months,
        'price_smooth': args.price_smooth
    }

    months = [
        '2016-01-01_2016-01-31',
        '2016-02-01_2016-02-29',
        '2016-03-01_2016-03-31',
        '2016-04-01_2016-04-30',
        '2016-05-01_2016-05-31',
        '2016-06-01_2016-06-30',
        '2016-07-01_2016-07-31',
        '2016-08-01_2016-08-31',
        '2016-09-01_2016-09-30',
        '2016-10-01_2016-10-31',
        '2016-11-01_2016-11-30',
        '2017-01-01_2017-01-31',
        '2017-02-01_2017-02-28',
        '2017-03-01_2017-03-31',
        '2017-04-01_2017-04-30',
        '2017-05-01_2017-05-31',
        '2017-06-01_2017-06-30',
        '2017-07-01_2017-07-31',
        '2017-08-01_2017-08-31',
        '2017-09-01_2017-09-30',
        '2017-10-01_2017-10-31',
        '2017-11-01_2017-11-30',
        '2018-01-01_2018-01-31',
        '2018-02-01_2018-02-28',
        '2018-03-01_2018-03-31',
        '2018-04-01_2018-04-30',
        '2018-05-01_2018-05-31',
        '2018-06-01_2018-06-30',
        '2018-07-01_2018-07-31',
        '2018-08-01_2018-08-31',
        '2018-09-01_2018-09-30',
        '2018-10-01_2018-10-31',
        '2018-11-01_2018-11-30']

    train_months = months[:args.num_of_train_months]

    test_months = months[args.num_of_train_months:args.num_of_train_months + args.num_of_test_months]

    ac_dict = {0: 0, 1: 0.25, 2: 0.5, 3: 0.6, 4: 7, 5: 0.8, 6: 0.9, 7: 1,
               8: 1.1, 9: 1.2, 10: 1.3, 11: 1.4, 12: 1.5, 13: 1.75, 14: 2, 15: 2.25, 16: 2.5, 17: 2.75, 18: 3}

    # ac_dict = {0: 0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 0.9, 5: 1, 6: 1.1, 7: 1.25,
    #            8: 1.5, 9: 1.75, 10: 2, 11: 2.5, 12: 3, 13: 3.5, 14: 4, 15: 4.5, 16: 5}
    ac_dict = {0: 1, 1: 1.02, 2: 1.04, 3: 1.06, 4: 1.08, 5: 1.1, 6: 1.12, 7: 1.14, 8: 1.16, 9: 1.18, 10: 1.2}
    # ac_dict = {0: 1, 1: 1.02, 2: 1.04, 3: 1.06, 4: 1.08, 5: 1.1}
    # ac_dict = {0: 0.9, 1: 0.92, 2: 0.94, 3: 0.96, 4: 0.98, 5: 1, 6: 1.02, 7: 1.04, 8: 1.06, 9: 1.08, 10: 1.1}
    # ac_dict = {0: 1, 1: 1.1, 2: 1.2, 3: 1.3, 4: 1.4, 5: 1.5, 6: 1.6, 7: 1.7, 8: 1.8, 9: 1.9, 10: 2.0}
    # Please always set Elapsed Time and Remaining Inventory True, otherwise AC Model will break down
    # ob_dict = {
    #     'Elapsed Time': True,
    #     'Remaining Inventory': True,
    #     'Bid L4 VWAP': True,
    #     'Ask L4 VWAP': True,
    #     'Bid L8 VWAP': True,
    #     'Ask L8 VWAP': True,
    #     'Bid L4 Volume': True,
    #     'Ask L4 Volume': True,
    #     'Bid L8 Volume': True,
    #     'Ask L8 Volume': True
    # }
    ob_dict = {
        'Elapsed Time': True,
        'Remaining Inventory': True,
        # 'Bid Ask Spread 1': True,
        # 'Bid Ask Spread 2': True,
        # 'Bid Ask Spread 3': True,
        # 'Bid Ask Spread 4': True,
        # 'Bid Ask Spread 5': True,
        # 'Bid Ask Spread 6': True,
        # 'Bid Ask Spread 7': True,
        # 'Bid Ask Spread 8': True,
        # 'Bid Ask Spread 9': True,
        # 'Bid Ask Spread 10': True,
        'Bid Price 1': True,
        'Bid Price 2': True,
        'Bid Price 3': True,
        'Bid Price 4': True,
        'Bid Price 5': True,
        # 'Bid Price 6': True,
        # 'Bid Price 7': True,
        # 'Bid Price 8': True,
        # 'Bid Price 9': True,
        # 'Bid Price 10': True,
        'Bid Volume 1': True,
        'Bid Volume 2': True,
        'Bid Volume 3': True,
        'Bid Volume 4': True,
        'Bid Volume 5': True,
        # 'Bid Volume 6': True,
        # 'Bid Volume 7': True,
        # 'Bid Volume 8': True,
        # 'Bid Volume 9': True,
        # 'Bid Volume 10': True,
        'Ask Price 1': True,
        'Ask Price 2': True,
        'Ask Price 3': True,
        'Ask Price 4': True,
        'Ask Price 5': True,
        # 'Ask Price 6': True,
        # 'Ask Price 7': True,
        # 'Ask Price 8': True,
        # 'Ask Price 9': True,
        # 'Ask Price 10': True,
        'Ask Volume 1': True,
        'Ask Volume 2': True,
        'Ask Volume 3': True,
        'Ask Volume 4': True,
        'Ask Volume 5': True,
        # 'Ask Volume 6': True,
        # 'Ask Volume 7': True,
        # 'Ask Volume 8': True,
        # 'Ask Volume 9': True,
        # 'Ask Volume 10': True,
    }


    print("============================================================")
    print("Reinforcement Learning for Optimal Execution")
    print("============================================================")
    print("Ticker:                          ", args.ticker)
    print("Volume to Liquidate (%):         ", args.liquidate_volume)
    print("Agent:                           ", args.agent)
    print("Total Loop:                      ", args.total_loop)
    print("Batch Size:                      ", args.batch_size)
    print("Initial Learning Rate:           ", args.learning_rate)
    print("Final Exploration Probability:   ", args.explore_stop)
    print("Exploration Decay:               ", args.decay_rate)
    print("Target Network Update (step):    ", args.target_network_update)
    print("Replay Buffer Size:              ", args.memory_size)
    print("Network Update Period (step):    ", args.network_update)
    print("LSTM Lookback:                   ", args.lstm_lookback)
    print("Number of Train Months:          ", args.num_of_train_months)
    print("Number of Test Months:           ", args.num_of_test_months)
    print("Price Smooth:                    ", args.price_smooth)
    print("============================================================")
    print("Observation Space:               ", ob_dict)
    print("Action Space:                    ", ac_dict)
    print("============================================================")

    if args.agent == 'dddqn' or args.agent == 'DDDQN':
        dddqn_train(hyperparameters, ac_dict, ob_dict, train_months, test_months)

    # if args.agent == 'dpg' or args.agent == 'DPG':
    #     DPG_Train(scenario_args, observation_space_args,
    #               action_space_args, reward_args, data_args, almgren_chriss_args)
    # elif args.agent == 'dqn' or args.agent == 'DQN':
    #     DQNTrain(scenario_args, observation_space_args,
    #              action_space_args, reward_args, data_args, almgren_chriss_args)
    # elif args.agent == 'drqn' or args.agent == 'DRQN':
    #     DRQNTrain(scenario_args, observation_space_args,
    #               action_space_args, reward_args, data_args, almgren_chriss_args, args.double)
    # elif args.agent == 'almgren_chriss':
    #     AlmgrenChrissTrain(scenario_args, observation_space_args,
    #                        action_space_args, reward_args, data_args, almgren_chriss_args)
    # elif args.agent == 'a2c' or args.agent == 'A2C':
    #     A2CTrain(scenario_args, observation_space_args,
    #              action_space_args, reward_args, data_args, almgren_chriss_args)
    # elif args.agent == 'dddqn' or args.agent == 'DDDQN':
    #     dddqn_train(scenario_args, observation_space_args)
    # else:
    #     raise Exception("Unknown Agent!")
