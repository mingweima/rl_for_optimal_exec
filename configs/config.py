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
    parser.add_argument('--total_loop', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--explore_stop', type=float, default=0.05)
    parser.add_argument('--decay_rate', type=float, default=0.04)
    parser.add_argument('--loop_update', type=int, default=3)
    parser.add_argument('--memory_size', type=int, default=200000)
    parser.add_argument('--network_update', type=int, default=30)

    args = parser.parse_args()
    hyperparameters = {
        'total_loop': args.total_loop,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'explore_stop': args.explore_stop,
        'decay_rate': args.decay_rate,
        'loop_update': args.loop_update,
        'memory_size': args.memory_size,
        'network_update': args.network_update
    }

    train_months = ['2016-01-01_2016-01-31',
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
                    '2016-12-01_2016-12-31',
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
                    '2017-12-01_2017-12-31',
                    '2018-01-01_2018-01-31',
                    '2018-02-01_2018-02-28',
                    '2018-03-01_2018-03-31',
                    '2018-04-01_2018-04-30',
                    '2018-05-01_2018-05-31',
                    '2018-06-01_2018-06-30']

    test_months = ['2018-07-01_2018-07-31',
                   '2018-08-01_2018-08-31',
                   '2018-09-01_2018-09-30',
                   '2018-10-01_2018-10-31',
                   '2018-11-01_2018-11-30',
                   '2018-12-01_2018-12-31']

    ac_dict = {0: 0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1, 5: 1.25,
               6: 1.5, 7: 1.75, 8: 2}

    # Please always set Elapsed Time and Remaining Inventory True, otherwise AC Model will break down
    ob_dict = {
        'Elapsed Time': True,
        'Remaining Inventory': True,
        'Bid Ask Spread 1': True,
        'Bid Ask Spread 2': True,
        'Bid Ask Spread 3': True,
        'Bid Ask Spread 4': True,
        'Bid Ask Spread 5': True,
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
    print("Agent:                           ", args.agent)
    print("Total Loop:                      ", args.total_loop)
    print("Batch Size:                      ", args.batch_size)
    print("Initial Learning Rate:           ", args.learning_rate)
    print("Final Exploration Probability:   ", args.explore_stop)
    print("Exploration Decay:               ", args.decay_rate)
    print("Target Network Update Period:    ", args.loop_update)
    print("Replay Buffer Size:              ", args.memory_size)
    print("Network Update Period (episode): ", args.network_update)
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
