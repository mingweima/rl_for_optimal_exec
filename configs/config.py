import argparse

from Agents.dqn.dqn_train import DQNTrain
from Agents.drqn.drqn_train import DRQNTrain
from Agents.dpg.dpg_train import DPG_Train
from Agents.a2c.a2c_train import A2CTrain
from Agents.almgren_chriss.almgren_chriss_train import AlmgrenChrissTrain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str)
    parser.add_argument('--reward', type=str, default='implementation shortfall')
    parser.add_argument('--time', type=int, default=60)
    parser.add_argument('--inventory', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--interval', type=int, default=5)
    args = parser.parse_args()

    ac_dict = {0: 0, 1: 0.01, 2: 0.02,
               3: 0.03, 4: 0.04, 5: 0.05,
               6: 0.06, 7: 0.07, 8: 0.08,
               9: 0.09, 10: 0.1, 11: 0.12,
               12: 0.14, 13: 0.16, 14: 0.18,
               15: 0.2, 16: 0.22, 17: 0.24,
               18: 0.26, 19: 0.28, 20: 0.3}
    
    ob_dict = {
        'Elapsed Time': True,
        'Remaining Inventory': True,
        'Bid Ask Spread': True,
        'Order Book Volume': True,
    }
    scenario_args = {
        'Time Horizon': args.time,
        'Initial Inventory': args.inventory,
        'Trading Interval': args.interval
    }

    observation_space_args = {
        'Observation Dictionary': ob_dict,
        'Upper Limit': 1,
        'Lower Limit': -1
    }
    action_space_args = {
        'Action Dictionary': ac_dict
    }

    reward_args = {
        'Reward Function': args.reward
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
        DPG_Train(scenario_args, observation_space_args, action_space_args, reward_args)
    elif args.agent == 'dqn' or args.agent == 'DQN':
        DQNTrain(scenario_args, observation_space_args, action_space_args, reward_args)
    elif args.agent == 'drqn' or args.agent == 'DRQN':
        DRQNTrain(scenario_args, observation_space_args, action_space_args, reward_args)
    elif args.agent == 'almgren_chriss':
        AlmgrenChrissTrain(scenario_args, observation_space_args, action_space_args, reward_args)
    elif args.agent == 'a2c' or args.agent == 'A2C':
        A2CTrain(scenario_args, observation_space_args, action_space_args, reward_args)
    else:
        raise Exception("Unknown Agent!")