from gym_simulator.envs.OrderBook import OrderBook
from gym_simulator.envs.OrderBookOracle import OrderBookOracle

import gym
from gym import spaces
import pandas as pd
import numpy as np

historical_date = pd.to_datetime('2019-06-03')
mkt_open = historical_date + pd.to_timedelta('09:30:00')
mkt_close = historical_date + pd.to_timedelta('16:00:00')
orders_file_path = 'E:/Git/rl_abmnew1/Market_Replay/gym_simulator/envs/sample_orders_file.csv'

class Simulator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Simulator, self).__init__()
        self.OrderBook = OrderBook()
        self.OrderBookOracle = OrderBookOracle(mkt_open, mkt_close, orders_file_path)
        self.current_step = 0
        self.inventory = 300
        # Action Space: Size of the Market Order (Buy: > 0, Sell: < 0)
        self.action_space = spaces.Box(
            low=-200, high=200, dtype=np.int64)
        # Observation Space: [Time, Inventory, Current_Price, Bid_Ask_Spread]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([100, 100, 100, 100]), dtype=np.float16)

    def reset(self):
        self.OrderBook = OrderBook()
        self.OrderBookOracle = OrderBookOracle(mkt_open, mkt_close, orders_file_path)
        self.current_step = 0
        self.inventory = 300
        return self.observation()


    def step(self, action):
        time = mkt_open + pd.to_timedelta('{}ms'.format(self.current_step))

        while OrderBookOracle.orders_list[0]['TIMESTAMP'] == time:
            self.OrderBook.handleLimitOrder(OrderBookOracle.orders_list[0])
            OrderBookOracle.orders_list.pop(0)

        execution_price, implementation_shortfall = self.OrderBook.handleMarketOrder(action)
        self.inventory += action
        self.current_step += 1
        done = self.inventory <= 0
        reward = -implementation_shortfall
        obs = self.observation()

        return obs, reward, done, {}

    def observation(self):
        return [self.current_step, self.inventory,
                self.OrderBook.getMidPrice(), self.OrderBook.getBidAskSpread()]

    def render(self, mode='human', close=False):
        print('Step: {}'.format(self.current_step))
        print('Inventory: {}'.format(self.inventory))