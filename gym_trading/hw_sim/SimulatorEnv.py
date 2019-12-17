import random
import numpy as np
import gym
from gym import spaces

from gym_trading.hw_sim.OrderBook import OrderBook
from gym_trading.hw_sim.config import ORDER_BOOK_ORACLE, MKT_OPEN
from gym_trading.hw_sim.AlmgrenChriss import AlmgrenChrissAgent

class Simulator(gym.Env):
    """
    The gym environment for optimal execution: managing a limit order book by taking in historical
    orders and reacting to the actions of the agent.
    """

    def __init__(self,
                 scenario_args,
                 observation_space_args,
                 action_space_args):
        super(Simulator, self).__init__()

        self.time_horizon = scenario_args['Time Horizon']
        self.initial_inventory = scenario_args['Initial Inventory']

        # Initializes the action space
        self.ac_dict = action_space_args['Action Dictionary']
        self.action_space = spaces.Discrete(len(self.ac_dict))

        # Initializes the observation space
        self.ob_dict = {k:v for k,v in observation_space_args['Observation Dictionary'].items() if v}
        self.observation_space = spaces.Box(
            low=observation_space_args['Lower Limit'],
            high=observation_space_args['Upper Limit'],
            shape=(len(self.ob_dict), 1),
            dtype=np.float64
        )

        # Initializes the baseline agent
        self.ac_agent = AlmgrenChrissAgent(time_horizon=self.time_horizon, sigma=0)

        # Initializes the Oracle by inputing historical data files.
        self.OrderBookOracle = ORDER_BOOK_ORACLE

    def reset(self):
        """
        Reset the environment before the start of the experient or after finishing one trial.

            Args:
                nothing
            Returns:
                obs (ndarray): the current observation
        """
        self.initial_time = random.randint(MKT_OPEN + 10, MKT_OPEN + 10)

        # Initialize the OrderBook
        self.OrderBook = OrderBook(self.OrderBookOracle.getHistoricalOrderBook(self.initial_time))

        self.current_time = self.initial_time
        self.initial_price = self.OrderBook.getMidPrice()
        self.inventory = self.initial_inventory
        obs = self.observation()
        self.ac_agent.reset()
        self.remaining_inventory_list = [self.initial_inventory]
        self.action_list = []
        self.indx_in_orders_list = 0

        return obs

    def step(self, action):
        """
        Placing an order into the limit order book according to the action

            Args:
                action (float64):
                    the proportion of the inventory on hand to execute at this time step (from 0 to 1)
            Returns:
                obs (ndarray): an ndarray specifying the limit order book
                reward (float64): the reward of this step
                done (boolean): whether this trajectory has ended or not
                info: any additional info
        """

        # Append the action taken to the action list
        self.action_list.append(action)

        # The action an Almgren Chriss Agent should take under the current condition
        ac_action = self.ac_agent.act(self.inventory / self.initial_inventory)

        # Add the market replay orders to the limit order book.
        while True:
            order = self.OrderBookOracle.orders_list[self.indx_in_orders_list]
            if order['TIME'] == self.current_time:
                self.OrderBook.handleLimitOrder(order)
            self.indx_in_orders_list += 1
            if order['TIME'] > self.current_time:
                break

        # Place the agent's order to the limit order book
        if self.current_time == self.initial_time + self.time_horizon:
            order_size = -self.inventory
        else:
            action = self.ac_dict[action]
            order_size = - round(self.inventory * action)
            if self.inventory + order_size < 0:
                order_size = - self.inventory
        if order_size != 0:
            vwap, _ = self.OrderBook.handleMarketOrder(order_size)
        else:
            vwap = 0

        # Calculate the reward by adding the regularization term to the execution shortfall
        ac_regularizor = - 0.1 * (- order_size - ac_action * self.inventory)**2
        shortfall = (-order_size) * (vwap - self.initial_price) / 10000
        reward = shortfall + 0 * ac_regularizor/ (1e4)

        # Update the environment and get new observation
        self.inventory += order_size
        self.remaining_inventory_list.append(self.inventory)
        done = self.inventory <= 0
        if done:
            self.ac_agent.reset()
        obs = self.observation()
        self.current_time += 1
        info = {'shortfall': shortfall}
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
            obs.append(self.OrderBook.getBidAskSpread()/10000)
        if 'Order Book Volume' in self.ob_dict.keys():
            obs.append(self.OrderBook.getBidAskVolume()/1000)
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