import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                               lamb=almgren_chriss_args['lamb'],
                               kappa=almgren_chriss_args['kappa'])

        # Initialize the Oracle by imputing historical data files.
        self.OrderBookOracle = OrderBookOracle(data_args, self.trading_interval)
        self.unique_date = self.OrderBookOracle.unique_date
        self.date = self.unique_date[-10:]

        # Only in use for env.render()
        self.normalized_price_list = []
        self.mid_price_list = []
        self.normalized_volume_list = []
        self.volume_list = []
        self.size_list = []
        self.reward_list = []

        # Only in use for fixed normalization
        self.price_mean, self.price_std, self.volume_mean, self.volume_std = \
            self.OrderBookOracle.get_past_price_volume(1)

    def reset(self):
        """
        Reset the environment before the start of the experiment or after finishing one trial.

            Returns:
                obs (nd array): the current observation
        """

        # Initialize the OrderBook
        initial_date = random.choice(self.unique_date[-10:])

        # initial_date = self.date[0]
        # self.date = self.date[1:]

        # self.price_mean, self.price_std, self.volume_mean, self.volume_std = \
        #     self.OrderBookOracle.get_past_price_volume(7, 5, initial_date)

        self.initial_time = initial_date + pd.Timedelta('11hours')
        self.current_time = self.initial_time
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(self.current_time))

        self.arrival_price = self.OrderBook.getMidPrice()
        self.remaining_inventory_list = []
        self.action_list = []

        self.inventory = self.initial_inventory
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

        self.normalized_price_list.append((self.OrderBook.getMidPrice() - self.price_mean) / self.price_std)
        self.normalized_volume_list.append((self.OrderBook.getBidsQuantity(1) - self.volume_mean) / self.volume_std)
        self.mid_price_list.append(self.OrderBook.getMidPrice())
        self.volume_list.append(self.OrderBook.getBidsQuantity(1))

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

        self.size_list.append(-order_size)

        implementation_shortfall = - (order_size / self.initial_inventory) * (vwap - self.arrival_price)

        # Calculate the reward
        if self.reward_function == 'implementation_shortfall':
            reward = implementation_shortfall
        elif self.reward_function == 'regularized_implementation_shortfall':
            ac_regularizor = - (- order_size - ac_action * self.inventory) ** 2
            reward = implementation_shortfall + ac_regularizor * 1e-5
        else:
            raise Exception("Unknown Reward Function!")

        self.reward_list.append(reward)

        # Update the environment and get new observation
        self.inventory += order_size
        self.remaining_inventory_list.append(self.inventory)

        done = (self.inventory <= 0)
        if done:
            self.ac_agent.reset()

        info = {'time': self.current_time,
                'shortfall': implementation_shortfall,
                'size': - order_size,
                'price_before_action': self.last_market_price}

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

            Returns:
                a vector reflecting the current market condition
        """
        obs = []
        if 'Elapsed Time' in self.ob_dict.keys():
            obs.append((self.current_time - self.initial_time) / self.time_horizon)
        if 'Remaining Inventory' in self.ob_dict.keys():
            obs.append(self.inventory / self.initial_inventory)
        for i in np.arange(1, 11):
            if 'Bid Price {}'.format(i) in self.ob_dict.keys():
                obs.append((self.OrderBook.getBidsPrice(i) - self.price_mean) / self.price_std)
            if 'Ask Price {}'.format(i) in self.ob_dict.keys():
                obs.append((self.OrderBook.getAsksPrice(i) - self.price_mean) / self.price_std)
            if 'Bid Volume {}'.format(i) in self.ob_dict.keys():
                obs.append((self.OrderBook.getBidsQuantity(i) - self.volume_mean) / self.volume_std)
            if 'Ask Volume {}'.format(i) in self.ob_dict.keys():
                obs.append((self.OrderBook.getAsksQuantity(i) - self.volume_mean) / self.volume_std)

        return np.asarray(obs)

    def render(self, mode='human', close=False):

        fig = plt.figure()
        volume1 = fig.add_subplot(221)
        volume1.plot(range(len(self.normalized_volume_list)), self.normalized_volume_list)
        volume1.set_ylim([-10, 10])
        volume1.set_title('Bids Level 1 volume for 10 days')
        volume2 = volume1.twinx()
        volume2.plot(range(len(self.normalized_volume_list)), self.volume_list, color='r', linestyle='dashed')
        volume2.set_ylim([0, 10000])

        price1 = fig.add_subplot(222)
        price1.plot(range(len(self.normalized_price_list)), self.normalized_price_list)
        price1.set_ylim([-2, 3])
        price1.set_title('Price for 10 days')
        price2 = price1.twinx()
        price2.plot(range(len(self.normalized_price_list)), self.mid_price_list, color='r', linestyle='dashed')
        price2.set_ylim([1840, 1940])

        re1 = fig.add_subplot(223)
        re1.bar(range(len(self.size_list)), self.size_list)
        re1.set_ylim([0, 10000])
        re1.set_title('Reward for 10 days')
        re2 = re1.twinx()
        re2.plot(range(len(self.size_list)), self.reward_list, self.reward_list, color='r', linestyle='dashed')
        re2.set_ylim([-1, 1])
        plt.show()

        plt.show()
