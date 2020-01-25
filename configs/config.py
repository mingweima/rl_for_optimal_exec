import argparse

import matplotlib.pyplot as plt

from agents.dqn.dqn_train import DQNTrain
from agents.drqn.drqn_train import DRQNTrain
from agents.dpg.dpg_train import DPG_Train
from agents.a2c.a2c_train import A2CTrain
from agents.almgren_chriss.almgren_chriss_train import AlmgrenChrissTrain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str)
    parser.add_argument('--reward', type=str, default='implementation_shortfall')
    parser.add_argument('--time', type=int, default=18000)
    parser.add_argument('--inventory', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--interval', type=int, default=600)
    parser.add_argument('--data', type=str, default='sample.csv')
    parser.add_argument('--eta', type=float, default=1.2256)
    parser.add_argument('--rho', type=float, default=0.1226)
    parser.add_argument('--sigma', type=float, default=0.257)
    parser.add_argument('--lamb', type=float, default=1e-4)
    parser.add_argument('--kappa', type=float, default=1e-3)
    parser.add_argument('--action_type', type=str, default='vanilla6')
    parser.add_argument('--hothead', type=str, default='False')
    parser.add_argument('--double', type=str, default='True')

    args = parser.parse_args()

    if args.action_type == 'prop_of_ac':
        ac_dict = {k: 0.1*k for k in range(21)}
    elif args.action_type == 'vanilla20':
        ac_dict = {0: 0, 1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.1,
                   6: 0.12, 7: 0.14, 8: 0.18, 9: 0.22, 10: 0.26, 11: 0.3,
                   12: 0.35, 13: 0.4, 14: 0.45, 15: 0.5, 16: 0.6, 17: 0.7, 18: 0.8, 19: 0.9, 20: 1}
    elif args.action_type == 'vanilla6':
        ac_dict = {0: 1, 1: 0.5, 2: 0.2, 3: 0.1, 4: 0.05, 5: 0}
            # 0: 0, 1: 0.05, 2: 0.1, 3: 0.2, 4: 0.5, 5: 1}
    else:
        raise Exception('Unknown Action Type')

    # Please always set Elapsed Time and Remaining Inventory True, otherwise AC Model will break down
    ob_dict = {
        'Elapsed Time': True,
        'Remaining Inventory': True,
        'Bid Price 1': True,
        'Bid Price 2': True,
        'Bid Price 3': True,
        'Bid Price 4': True,
        'Bid Volume 1': True,
        'Bid Volume 2': True,
        'Bid Volume 3': True,
        'Bid Volume 4': True,
        'Ask Price 1': True,
        'Ask Price 2': True,
        'Ask Price 3': True,
        'Ask Price 4': True,
        'Ask Volume 1': True,
        'Ask Volume 2': True,
        'Ask Volume 3': True,
        'Ask Volume 4': True,
    }
    scenario_args = {
        'Time Horizon': args.time,
        'Initial Inventory': args.inventory,
        'Trading Interval': args.interval,
        'Hothead': args.hothead
    }

    observation_space_args = {
        'Observation Dictionary': ob_dict,
        'Upper Limit': 1,
        'Lower Limit': -1
    }
    action_space_args = {
        'Action Type': args.action_type,
        'Action Dictionary': ac_dict
    }

    reward_args = {
        'Reward Function': args.reward
    }

    data_args = args.data

    almgren_chriss_args = {
        'eta': args.eta,
        'rho': args.rho,
        'sigma': args.sigma,
        'lamb': args.lamb,
        'kappa': args.kappa
    }

    print("============================================================")
    print("Reinforcement Learning for Optimal Execution")
    print("============================================================")
    print("Time Horizon (Seconds):     ", args.time)
    print("Trading Interval (Seconds): ", args.interval)
    print("Initial Inventory:          ", args.inventory)
    print("Agent:                      ", args.agent)
    print("============================================================")

    if args.agent == 'dpg' or args.agent == 'DPG':
        DPG_Train(scenario_args, observation_space_args,
                  action_space_args, reward_args, data_args, almgren_chriss_args)
    elif args.agent == 'dqn' or args.agent == 'DQN':
        DQNTrain(scenario_args, observation_space_args,
                 action_space_args, reward_args, data_args, almgren_chriss_args)
    elif args.agent == 'drqn' or args.agent == 'DRQN':
        DRQNTrain(scenario_args, observation_space_args,
                  action_space_args, reward_args, data_args, almgren_chriss_args, args.double)
    elif args.agent == 'almgren_chriss':
        AlmgrenChrissTrain(scenario_args, observation_space_args,
                           action_space_args, reward_args, data_args, almgren_chriss_args)
    elif args.agent == 'a2c' or args.agent == 'A2C':
        A2CTrain(scenario_args, observation_space_args,
                 action_space_args, reward_args, data_args, almgren_chriss_args)
    else:
        raise Exception("Unknown Agent!")



    # scenario_args['Hothead'] = 'False'
    # almgren_chriss_args['sigma'] = 0
    # almgren_chriss_args['lamb'] = 0
    # linear_reward = AlmgrenChrissTrain(scenario_args, observation_space_args,
    #                                     action_space_args, reward_args, data_args, almgren_chriss_args)
    #
    # scenario_args['Hothead'] = 'False'
    # almgren_chriss_args['kappa'] = 0.0005
    # ac_reward1 = AlmgrenChrissTrain(scenario_args, observation_space_args,
    #                                     action_space_args, reward_args, data_args, almgren_chriss_args)
    #
    #
    # almgren_chriss_args['kappa'] = 0.001
    # ac_reward2 = AlmgrenChrissTrain(scenario_args, observation_space_args,
    #                                     action_space_args, reward_args, data_args, almgren_chriss_args)
    #
    # almgren_chriss_args['kappa'] = 0.005
    # ac_reward3 = AlmgrenChrissTrain(scenario_args, observation_space_args,
    #                                     action_space_args, reward_args, data_args, almgren_chriss_args)
    #
    # almgren_chriss_args['kappa'] = 0.01
    # ac_reward4 = AlmgrenChrissTrain(scenario_args, observation_space_args,
    #                                     action_space_args, reward_args, data_args, almgren_chriss_args)
    #
    # scenario_args['Hothead'] = 'True'
    # hothead_reward = AlmgrenChrissTrain(scenario_args, observation_space_args,
    #                                     action_space_args, reward_args, data_args, almgren_chriss_args)
    #
    # labels = ['Linear', 'AC (5e-4)', 'AC (1e-3)', 'AC (5e-3)', 'AC (1e-2)', 'Hothead']
    # plt.boxplot([linear_reward, ac_reward1, ac_reward2, ac_reward3, ac_reward4, hothead_reward], labels=labels)
    # plt.show()
