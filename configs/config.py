import argparse

from Agents.dqn.dqn_train import DQNTrain
from Agents.drqn.drqn_train import DRQNTrain
from Agents.dpg.dpg_train import DPG_Train
from Agents.a2c.a2c_train import A2CTrain
from Agents.almgren_chriss.almgren_chriss_train import AlmgrenChrissTrain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str)
    parser.add_argument('--reward', type=str, default='implementation_shortfall')
    parser.add_argument('--time', type=int, default=600)
    parser.add_argument('--inventory', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--data', type=str, default='sample.csv')
    parser.add_argument('--eta', type=int, default=1.2256)
    parser.add_argument('--rho', type=int, default=0.1226)
    parser.add_argument('--sigma', type=int, default=6.67e-4)
    parser.add_argument('--lamb', type=int, default=1e-4)
    parser.add_argument('--action_type', type=str, default='prop_of_ac')
    parser.add_argument('--hothead', type=str, default='False')
    args = parser.parse_args()

    if args.action_type == 'prop_of_ac':
        ac_dict = {k:0.1*k for k in range(21)}
    elif args.action_type == 'vanilla_action':
        ac_dict = {0: 0, 1: 0.02, 2: 0.04,3: 0.06, 4: 0.08, 5: 0.1,
                   6: 0.12, 7: 0.14, 8: 0.18, 9: 0.22, 10: 0.26, 11: 0.3,
                   12: 0.35, 13: 0.4, 14: 0.45, 15: 0.5, 16: 0.6, 17: 0.7, 18: 0.8, 19: 0.9, 20: 1}
    else:
        raise Exception('Unknown Action Type')

    # Please always set Elapsed Time and Remaining Inventory True, otherwise AC Model will break down
    ob_dict = {
        'Elapsed Time': True,
        'Remaining Inventory': True,
        'Bid Ask Spread': False,
        'Order Book Volume': False,
        'Market Price': False,
        'Log Return': False
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
        'lamb': args.lamb
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
                  action_space_args, reward_args, data_args, almgren_chriss_args)
    elif args.agent == 'almgren_chriss':
        AlmgrenChrissTrain(scenario_args, observation_space_args,
                           action_space_args, reward_args, data_args, almgren_chriss_args)
    elif args.agent == 'a2c' or args.agent == 'A2C':
        A2CTrain(scenario_args, observation_space_args,
                 action_space_args, reward_args, data_args, almgren_chriss_args)
    else:
        raise Exception("Unknown Agent!")