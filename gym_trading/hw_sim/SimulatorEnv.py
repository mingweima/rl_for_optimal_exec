import random

import pandas as pd
import numpy as np
import gym
from gym import spaces

from gym_trading.hw_sim.OrderBook import OrderBook
from gym_trading.hw_sim.config import ORDER_BOOK_ORACLE, MKT_OPEN
from gym_trading.hw_sim.AlmgrenChriss import AlmgrenChrissAgent
from gym_trading.hw_sim.data_OMI.OMI_Data_Oracle import OrderBookOracle


class Simulator(gym.Env):
    """
    The gym environment for optimal execution: managing a limit order book by taking in historical
    orders and reacting to the actions of the agent.
    """

    def __init__(self,
                 scenario_args,
                 observation_space_args,
                 action_space_args,
                 reward_args,
                 data_args,
                 almgren_chriss_args):
        super(Simulator, self).__init__()

        self.hothead = scenario_args['Hothead']
        self.trading_interval = scenario_args['Trading Interval']
        self.time_horizon = pd.Timedelta(seconds=scenario_args['Time Horizon'])
        self.initial_inventory = scenario_args['Initial Inventory']

        # Initialize the action space
        self.ac_type = action_space_args['Action Type']
        self.ac_dict = action_space_args['Action Dictionary']
        self.action_space = spaces.Discrete(len(self.ac_dict))

        # Initialize the observation space
        self.ob_dict = {k:v for k,v in observation_space_args['Observation Dictionary'].items() if v}
        self.observation_space = spaces.Box(
            low=observation_space_args['Lower Limit'],
            high=observation_space_args['Upper Limit'],
            shape=(len(self.ob_dict), 1),
            dtype=np.float64
        )

        # Initialize the reward function
        self.reward_function = reward_args['Reward Function']

        # Initialize the baseline agent
        self.ac_agent = AlmgrenChrissAgent(
                               ac_type=self.ac_type,
                               ac_dict=self.ac_dict,
                               time_horizon=scenario_args['Time Horizon'],
                               eta=almgren_chriss_args['eta'],
                               rho=almgren_chriss_args['rho'],
                               sigma=almgren_chriss_args['sigma'],
                               tau=scenario_args['Trading Interval'],
                               lamb=almgren_chriss_args['lamb'])

        # Initialize the Oracle by inputing historical data files.
        self.OrderBookOracle = OrderBookOracle(data_args, self.trading_interval)

    def reset(self):
        """
        Reset the environment before the start of the experient or after finishing one trial.

            Args:
                nothing
            Returns:
                obs (ndarray): the current observation
        """

        def sample_initial_time():
            day = random.choice([2, 3, 4, 7])
            lower_limit = pd.to_datetime('2013/1/{} 09:00:00'.format(day))
            upper_limit = pd.to_datetime('2013/1/{} 16:00:00'.format(day))
            time = lower_limit + random.random() * (upper_limit - lower_limit)
            time.round('{}s'.format(self.trading_interval))
            return time

        # Initialize the OrderBook
        # self.initial_time = sample_initial_time()
        self.initial_time = pd.to_datetime('2013/1/2 09:00:00')
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(self.initial_time))
        self.initial_price = self.OrderBook.getMidPrice()

        self.current_time = self.initial_time + pd.Timedelta(seconds=self.trading_interval)
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(self.current_time))

        self.remaining_inventory_list = []
        self.action_list = []
        self.inventory = self.initial_inventory
        self.last_market_price = self.initial_price
        self.ac_agent.reset()

        return self.observation()

    def step(self, action):
        """
        Placing an order into the limit order book according to the action

            Args:
                action (int64): from 0 to (ac_dim - 1)
            Returns:
                obs (ndarray): an ndarray specifying the limit order book
                reward (float64): the reward of this step
                done (boolean): whether this trajectory has ended or not
                info: any additional info
        """

        # Set the last market price before taking any action and updating the LOB
        self.last_market_price = self.OrderBook.getMidPrice()

        # Append the action taken to the action list
        self.action_list.append(action)

        # The action an Almgren Chriss Agent should take under the current condition
        ac_action = self.ac_agent.act(self.inventory / self.initial_inventory)

        # Place the agent's order to the limit order book
        if self.hothead == 'True':
            order_size = - self.inventory
        elif self.current_time + pd.Timedelta(seconds=self.trading_interval) > self.initial_time + self.time_horizon:
            order_size = -self.inventory
        else:
            if self.ac_type == 'vanilla_action':
                action = self.ac_dict[action]
                order_size = - round(self.inventory * action)
            elif self.ac_type == 'prop_of_ac':
                action = self.ac_dict[action]
                order_size = - round(action * ac_action * self.inventory)
            else:
                raise Exception('Unknown Action Type')
            if self.inventory + order_size < 0:
                order_size = - self.inventory

        if order_size != 0:
            vwap, _ = self.OrderBook.handleMarketOrder(order_size)
        else:
            vwap = 0

        implementation_shortfall = - order_size * (vwap - self.initial_price)


        # Calculate the reward
        if self.reward_function == 'implementation_shortfall':
            reward = implementation_shortfall
        elif self.reward_function == 'regularized_implementation_shortfall':
            ac_regularizor = - (- order_size - ac_action * self.inventory) ** 2
            reward = implementation_shortfall + ac_regularizor * 1e-5
        else:
            raise Exception("Unknown Reward Function!")

        # Update the environment and get new observation
        self.inventory += order_size
        self.remaining_inventory_list.append(self.inventory)

        done = (self.inventory <= 0)
        if done:
            self.ac_agent.reset()

        info = {'shortfall': implementation_shortfall}

        # Update the time
        self.current_time += pd.Timedelta(seconds=self.trading_interval)

        # Update the LOB. (for OMI data)
        self.OrderBook.update(self.OrderBookOracle.getHistoricalOrderBook(self.current_time))

        # Take an observation of the current state
        obs = self.observation()
        return obs, reward, done, info

    def observation(self):
        """
        Take an observation of the current environment

            Args:
                 none
            Returns:
                a vector reflecting the current market condition
        """
        obs = []
        if 'Elapsed Time' in self.ob_dict.keys():
            obs.append((self.current_time - self.initial_time)/self.time_horizon)
        if 'Remaining Inventory' in self.ob_dict.keys():
            obs.append(self.inventory/self.initial_inventory)
        if 'Bid Ask Spread' in self.ob_dict.keys():
            obs.append(self.OrderBook.getBidAskSpread())
        if 'Order Book Volume' in self.ob_dict.keys():
            obs.append(self.OrderBook.getBidAskVolume())
        if 'Log Return' in self.ob_dict.keys():
            obs.append(np.log(self.OrderBook.getMidPrice()/self.last_market_price))
        if 'Market Price' in self.ob_dict.keys():
            obs.append(np.log(self.OrderBook.getMidPrice()/self.initial_price))

        return np.asarray(obs)

    def render(self, mode='human', close=False):
        """
        Print the agent's current holdings and some key market parameters
        """
        print('Inventory: {}'.format(self.inventory))
        print('Time: ', self.current_time, 'Price: ', self.OrderBook.getMidPrice(),
              'Asks: ', self.OrderBook.getAsksQuantity(),
              'Bids: ', self.OrderBook.getBidsQuantity())
        print('Asks: ', self.OrderBook.getInsideAsks())
        print('Bids: ', self.OrderBook.getInsideBids(), '\n')
        print("Remaining Inventory List: ", self.remaining_inventory_list)
        print("Action List: ", self.action_list)