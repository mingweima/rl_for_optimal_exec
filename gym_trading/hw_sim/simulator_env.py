from gym_trading.hw_sim.OrderBook import OrderBook
from gym_trading.hw_sim.OrderBookOracle import OrderBookOracle
from gym_trading.hw_sim.config import ORDER_BOOK_ORACLE, MKT_OPEN

import random
import gym
from gym import spaces
import numpy as np
import os


class Simulator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Simulator, self).__init__()
        self.current_time = MKT_OPEN + 1
        self.time_horizon = 50
        self.num_of_spread_state = 10
        self.num_of_volume_state = 10
        # Initializes the Oracle by inputing historical data files.
        self.OrderBookOracle = ORDER_BOOK_ORACLE
        # Initializes the OrderBook at a given historical time.
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(MKT_OPEN + 1))
        # Inventory of shares hold to sell.
        self.initial_inventory = 100
        # Action Space
        self.action_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float64)
        # Observation Space: [Time, Inventory, Spread State, Volume State]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]), dtype=np.float16)


    def reset(self):
        self.initial_time = random.randint(MKT_OPEN+100, MKT_OPEN+100)
        self.current_time = self.initial_time
        self.OrderBookOracle = ORDER_BOOK_ORACLE
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
            action = 1
        else:

            # Replacement = {0: 0, 1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.1, 6: 0.2, 7: 0.25, 8: 0.5, 9: 1}
            # order_size = -round(self.inventory * Replacement[action])
            order_size = self.inventory * action
            order_size = -round(order_size)


        if order_size != 0:
            # print(order_size)
            execution_price, implementation_shortfall = self.OrderBook.handleMarketOrder(order_size)
        else:
            execution_price, implementation_shortfall = 0, 0

        reward = -implementation_shortfall if action < 0.5 else -1e5 - implementation_shortfall

        self.inventory += order_size
        # print(self.inventory)
        done = self.inventory <= 0
        obs = self.observation()
        self.current_time += 1
        return obs, reward/10000, done, {}

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