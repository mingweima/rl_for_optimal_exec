from gym_simulator.envs.OrderBook import OrderBook
from gym_simulator.envs.OrderBookOracle import OrderBookOracle

import gym
from gym import spaces
import pandas as pd
import numpy as np

# mkt_open and mkt_close are in unit "second": 34200 denotes 09:30 and 57600 denotes 16:00
mkt_open = 34200
mkt_close = 57600
orders_file_path = 'E:/Git/rl_abmnew1/Market_Replay/gym_simulator' \
                   '/envs/GOOG_2012-06-21_34200000_57600000_message_5.csv'
LOB_file_path = 'E:/Git/rl_abmnew1/Market_Replay/gym_simulator' \
                '/envs/GOOG_2012-06-21_34200000_57600000_orderbook_5.csv'

class Simulator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Simulator, self).__init__()
        self.current_time = 34201

        # Initializes the Oracle by inputing historical data files.
        self.OrderBookOracle = OrderBookOracle(mkt_open, mkt_close, orders_file_path, LOB_file_path)
        # Initializes the OrderBook at a given historical time.
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(self.current_time))
        # Inventory of shares hold to sell.
        self.inventory = 300
        # Action Space: Size of the Market Order (Buy: > 0, Sell: < 0)
        self.action_space = spaces.Box(
            low=-200, high=200, shape=(1,), dtype=np.int64)
        # Observation Space: [Time, Inventory, Current_Price, Bid_Ask_Spread]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([100, 100, 100, 100]), dtype=np.float16)

    def reset(self):
        self.current_time = 34201
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(self.current_time))
        self.OrderBookOracle = OrderBookOracle(mkt_open, mkt_close, orders_file_path, LOB_file_path)
        self.inventory = 300


    def step(self, action):
        self.current_time += 1

        # Add market replay orders.
        while self.OrderBookOracle.orders_list[0]['TIME'] <= self.current_time:
            if self.OrderBookOracle.orders_list[0]['TIME'] == self.current_time:
                self.OrderBook.handleLimitOrder(self.OrderBookOracle.orders_list[0])
            self.OrderBookOracle.orders_list.pop(0)

        # Take action (market order) and calculate reward
        execution_price, implementation_shortfall = self.OrderBook.handleMarketOrder(action)
        reward = -implementation_shortfall

        self.inventory += action
        done = self.inventory <= 0
        obs = self.observation()

        return obs, reward, done, {}

    def observation(self):
        return [self.current_time, self.inventory,
                self.OrderBook.getInsideBids(), self.OrderBook.getInsideAsks()]

    def render(self, mode='human', close=False):
        # print('Step: {}'.format(self.current_step))
        # print('Inventory: {}'.format(self.inventory))
        print('Bids: \n', self.OrderBook.bids)
        print('Asks: \n', self.OrderBook.asks)