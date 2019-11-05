from gym_simulator.envs.OrderBook import OrderBook
from gym_simulator.envs.OrderBookOracle import OrderBookOracle

import random
import gym
from gym import spaces
import pandas as pd
import numpy as np

# mkt_open and mkt_close are in unit "second": 34200 denotes 09:30 and 57600 denotes 16:00
mkt_open = 34200
mkt_close = 57600
orders_file_path = 'E:/Git/rl_abmnew1/Market_Replay/gym_simulator/envs' \
                   '/AAPL_2012-06-21_34200000_57600000_message_10.csv'
LOB_file_path = 'E:/Git/rl_abmnew1/Market_Replay/gym_simulator/envs' \
                '/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv'

class Simulator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Simulator, self).__init__()
        self.current_time = mkt_open + 1
        self.time_horizon = 30
        self.num_of_spread_state = 10
        self.num_of_volume_state = 10
        # Initializes the Oracle by inputing historical data files.
        self.OrderBookOracle = OrderBookOracle(mkt_open, mkt_close, orders_file_path, LOB_file_path)
        # Initializes the OrderBook at a given historical time.
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(mkt_open + 1))
        # Inventory of shares hold to sell.
        self.initial_inventory = 500
        # Action Space: Size of the Market Order (Buy: > 0, Sell: < 0)
        self.action_space = spaces.Box(
            low=0, high=9, shape=(1,), dtype=np.int64)
        # Observation Space: [Time, Inventory, Spread State, Volume State]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]), dtype=np.float16)

    def reset(self):
        self.initial_time = random.randint(mkt_open+100, mkt_open+100)
        self.current_time = self.initial_time
        self.OrderBookOracle = OrderBookOracle(mkt_open, mkt_close, orders_file_path, LOB_file_path)
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(self.current_time - 1))
        self.inventory = self.initial_inventory
        obs = self.observation()
        return obs


    def step(self, action):

        # Add market replay orders.
        while self.OrderBookOracle.orders_list[0]['TIME'] <= self.current_time:
            if self.OrderBookOracle.orders_list[0]['TIME'] == self.current_time:
                self.OrderBook.handleLimitOrder(self.OrderBookOracle.orders_list[0])
            self.OrderBookOracle.orders_list.pop(0)

        # Take action (market order) and calculate reward
        if self.current_time == self.initial_time + self.time_horizon:
            order_size = -self.inventory
        else:
            Replacement = {0: 0, 1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.1, 6: 0.2, 7: 0.25, 8: 0.5, 9: 1}
            order_size = -round(self.inventory * Replacement[action])

        if order_size != 0:
            execution_price, implementation_shortfall = self.OrderBook.handleMarketOrder(order_size)
        else:
            execution_price, implementation_shortfall = 0, 0
        reward = -implementation_shortfall - 100

        self.inventory += order_size
        done = self.inventory <= 0
        obs = self.observation()
        self.current_time += 1
        return obs, reward, done, {}

    def observation(self):
        time_index = (self.current_time - self.initial_time)/100
        inventory_index = self.inventory/self.initial_inventory
        spread_index = self.OrderBook.getBidAskSpread()/10000
        volume_index = self.OrderBook.getBidAskVolume()/1000

        return [time_index, inventory_index, spread_index, volume_index]

    def render(self, mode='human', close=False):
        # print('Step: {}'.format(self.current_step))
        # print('Inventory: {}'.format(self.inventory))
        print('Time: ', self.current_time, 'Price: ', self.OrderBook.getMidPrice(),
              'Asks: ', self.OrderBook.getAsksQuantity(),
              'Bids: ', self.OrderBook.getBidsQuantity())
        print('Asks: ', self.OrderBook.getInsideAsks())
        print('Bids: ', self.OrderBook.getInsideBids(), '\n')