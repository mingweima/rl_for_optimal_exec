from gym_trading.hw_sim.OrderBook import OrderBook
from gym_trading.hw_sim.OrderBookOracle import OrderBookOracle
from gym_trading.hw_sim.config import ORDER_BOOK_ORACLE, MKT_OPEN
from gym_trading.hw_sim.AlmgrenChriss import AlmgrenChrissAgent

import random
import gym
from gym import spaces
import numpy as np
from copy import deepcopy


class Simulator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Simulator, self).__init__()
        # self.initial_time = random.randint(MKT_OPEN + 10, MKT_OPEN + 10)
        self.initial_time = MKT_OPEN + 10
        self.time_horizon = 20
        # Initializes the Oracle by inputing historical data files.
        self.OrderBookOracle = ORDER_BOOK_ORACLE
        self.InitialOrderBookOracle = ORDER_BOOK_ORACLE
        # Initializes the OrderBook at a given historical time.
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(self.initial_time - 1))
        self.InitialOrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(self.initial_time - 1))
        # Inventory of shares hold to sell.
        self.initial_inventory = 1000
        # Action Space
        self.action_space = spaces.Discrete(11)
        # Observation Space: [Time, Inventory, Spread State, Volume State]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]), dtype=np.float16)

        self.ac_num_to_act_dict = \
            {0: 0, 1: 0.01, 2: 0.02, 3: 0.05, 4: 0.08, 5: 0.1, 6: 0.2, 7: 0.3, 8: 0.4, 9: 0.5, 10: 1.0}
        self.ac_agent = AlmgrenChrissAgent(time_horizon=self.time_horizon, sigma=0)

    def reset(self):
        self.initial_time = random.randint(MKT_OPEN + 10, MKT_OPEN + 10)
        self.OrderBook = self.InitialOrderBook
        self.OrderBookOracle = self.InitialOrderBookOracle
        self.current_time = self.initial_time
        self.initial_price = self.OrderBook.getMidPrice()
        self.inventory = self.initial_inventory
        obs = self.observation()
        self.ac_agent.reset()
        self.accumulated_shortfall = 0
        self.remaining_inventory_list = []
        self.remaining_inventory_list.append(self.initial_inventory)
        self.action_list = []

        self.indx_in_orders_list = 0

        return obs

    def step(self, action):
        self.action_list.append(action)
        ac_action = self.ac_agent.act(self.inventory / self.initial_inventory)
        # Add market replay orders.
        while True:
            order = self.OrderBookOracle.orders_list[self.indx_in_orders_list]
            if order['TIME'] == self.current_time:
                self.OrderBook.handleLimitOrder(order)
            self.indx_in_orders_list += 1
            if order['TIME'] > self.current_time:
                break
        # while self.orders_list[0]['TIME'] <= self.current_time:
        #     if self.orders_list[0]['TIME'] == self.current_time:
        #         self.OrderBook.handleLimitOrder(self.orders_list[0])
        #     self.orders_list.pop(0)

        # Take action (market order) and calculate reward
        if self.current_time == self.initial_time + self.time_horizon:
            order_size = -self.inventory  # action = 1.0
        else:
            action = self.ac_num_to_act_dict[action]
            order_size = - round(self.inventory * action)
            if self.inventory + order_size < 0:
                order_size = - self.inventory

        self.current_price = self.OrderBook.getMidPrice()

        if order_size != 0:
            vwap, _ = self.OrderBook.handleMarketOrder(order_size)
        else:
            vwap = 0
        ac_regularizor = - 0.1 * (- order_size - ac_action * self.inventory)**2
        shortfall = (-order_size) * (vwap - self.initial_price) / 10000
        self.inventory += order_size
        self.remaining_inventory_list.append(self.inventory)
        done = self.inventory <= 0
        if done:
            self.ac_agent.reset()
        obs = self.observation()
        reward = 0 * shortfall + ac_regularizor/ (1e4)
        self.current_time += 1
        info = {'shortfall': shortfall}
        return obs, reward, done, info

    def observation(self):
        time_index = (self.current_time - self.initial_time)/self.time_horizon
        inventory_index = self.inventory/self.initial_inventory
        spread_index = self.OrderBook.getBidAskSpread()/10000
        volume_index = self.OrderBook.getBidAskVolume()/1000

        return np.asarray(([time_index, inventory_index, spread_index, volume_index]))

    def render(self, mode='human', close=False):
        print('Inventory: {}'.format(self.inventory))
        print('Time: ', self.current_time, 'Price: ', self.OrderBook.getMidPrice(),
              'Asks: ', self.OrderBook.getAsksQuantity(),
              'Bids: ', self.OrderBook.getBidsQuantity())
        print('Asks: ', self.OrderBook.getInsideAsks())
        print('Bids: ', self.OrderBook.getInsideBids(), '\n')
        print("Remaining Inventory List: ", self.remaining_inventory_list)
        print("Action List: ", self.action_list)
