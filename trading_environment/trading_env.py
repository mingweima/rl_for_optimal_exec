import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from trading_environment.orderbook import OrderBook

EPISODE_LENGTH_IN_SECONDS = 18000
TRADING_INTERVAL_IN_SECONDS = 600

class Simulator:
    """
    The gym trading_environment for optimal execution: managing a limit order book by taking in historical
    orders and reacting to the actions of the agent.
    """

    def __init__(self, data_dict, date_dict, ac_dict, ob_dict, initial_shares, look_back):
        self.data_dict = data_dict
        self.date_dict = date_dict

        self.hothead = 'False'
        self.trading_interval = TRADING_INTERVAL_IN_SECONDS
        self.time_horizon = pd.Timedelta(seconds=EPISODE_LENGTH_IN_SECONDS)
        self.trading_steps = int(self.time_horizon.seconds / self.trading_interval)
        self.initial_inventory = initial_shares
        self.look_back = look_back

        # Initialize the action space
        self.ac_dict = ac_dict
        self.ac_type = 'prop of linear'

        # Initialize the observation space
        self.ob_dict = {k: v for k, v in ob_dict.items() if v}

        # Initialize the reward function
        self.reward_function = 'implementation_shortfall'

    def reset(self, month, day):
        """
        Reset the trading_environment before the start of the experiment or after finishing one trial.

            Returns:
                obs (nd array): the current observation
        """

        # Initialize the OrderBook
        # initial_date = random.choice(self.unique_date[-10:])
        # initial_date = self.unique_date[num_days]
        self.data = self.data_dict[month][day]
        initial_date = day

        mid_price_list = []
        volume_list = []

        # 5 steps of normalization
        for interval in range(5):
            hour = 0
            while True:
                LOB = np.array(self.data.loc[self.data['Date-Time'] >=
                                             day + pd.Timedelta('{}hours'.format(11 - hour)) +
                                             pd.Timedelta('{}hours'.format(interval))].head(1))
                try:
                    LOB = LOB[0]
                    break
                except:
                    # print('Cannot find LOB for ', day + pd.Timedelta('{}hours'.format(11 - hour))
                    #       + pd.Timedelta('{}hours'.format(interval)))
                    # print('Use instead LOB for ', day + pd.Timedelta('{}hours'.format(11 - hour - 1))
                    #       + pd.Timedelta('{}hours'.format(interval)))
                    hour += 1

            mid_price = (LOB[1] + LOB[3]) / 2
            if mid_price:
                mid_price_list.append(mid_price)
            volume = (sum(LOB[4 * j + 2] for j in range(10)) + sum(LOB[4 * j + 4] for j in range(10))) / 20
            if volume:
                volume_list.append(volume)

        self.price_mean, self.price_std, self.volume_mean, self.volume_std = \
            np.average(mid_price_list), np.std(mid_price_list), np.average(volume_list), np.std(volume_list)

        # Only in use for env.render()
        self.normalized_price_list = []
        self.mid_price_list = []
        self.normalized_volume_list = []
        self.volume_list = []
        self.size_list = []
        self.reward_list = []

        self.inventory = self.initial_inventory

        self.initial_time = initial_date + pd.Timedelta('11hours')

        # Initialize the observation sequence
        self.observation_sequence = []
        self.current_time = self.initial_time - pd.Timedelta(seconds=self.trading_interval)
        self.OrderBook = OrderBook(self.get_historical_order())
        while self.current_time > initial_date + pd.Timedelta('8hours'):
            self.OrderBook.update(self.get_historical_order())
            self.observation_sequence.append(self.observation())
            self.current_time = self.current_time - pd.Timedelta(seconds=self.trading_interval)
        self.observation_sequence.reverse()

        self.current_time = self.initial_time
        self.OrderBook = OrderBook(self.get_historical_order())

        self.arrival_price = self.OrderBook.getMidPrice()
        self.remaining_inventory_list = []
        self.action_list = []

        return self.observation_sequence[-self.look_back:]

    def get_historical_order(self):
        hour = 0
        while True:
            LOB = np.array(
                self.data.loc[self.data['Date-Time'] >= self.current_time - pd.Timedelta('{}hours'.format(hour))].head(
                    1))
            try:
                LOB = LOB[0]
                break
            except:
                # print('Cannot find LOB for ', self.current_time - pd.Timedelta('{}hours'.format(hour)))
                # print('Use instead LOB for ', self.current_time - pd.Timedelta('{}hours'.format(hour + 1)))
                hour += 1

        bids = []
        for i in range(10):
            price = LOB[4 * i + 1]
            size = LOB[4 * i + 2]
            bids.append({'TIME': self.current_time,
                         'TYPE': 0, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'BUY'})
        asks = []
        for i in range(10):
            price = LOB[4 * i + 3]
            size = LOB[4 * i + 4]
            asks.append({'TIME': self.current_time,
                         'TYPE': 0, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'SELL'})

        return [bids, asks]

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
        # Update the time
        self.current_time += pd.Timedelta(seconds=self.trading_interval)

        # Update the LOB. (for OMI data)
        self.OrderBook.update(self.get_historical_order())

        # Take an observation of the current state
        obs = self.observation()
        self.observation_sequence.append(obs)

        # Set the last market price before taking any action and updating the LOB
        self.last_market_price = self.OrderBook.getMidPrice()

        # Append the action taken to the action list
        self.action_list.append(action)

        self.normalized_price_list.append((self.OrderBook.getMidPrice() - self.price_mean) / self.price_std)
        self.normalized_volume_list.append((self.OrderBook.getBidsQuantity(1) - self.volume_mean) / self.volume_std)
        self.mid_price_list.append(self.OrderBook.getMidPrice())
        self.volume_list.append(self.OrderBook.getBidsQuantity(1))

        # Place the agent's order to the limit order book
        if self.hothead == 'True':
            order_size = - self.inventory
        elif action == -1:
            order_size = - self.inventory
            print('wkfds')
        elif self.current_time + pd.Timedelta(seconds=self.trading_interval) > self.initial_time + self.time_horizon:
            order_size = -self.inventory
        else:
            if self.ac_type == 'vanilla':
                action = self.ac_dict[action]
                order_size = - round(self.initial_inventory * action)
            elif self.ac_type == 'prop of linear':
                action = self.ac_dict[action]
                order_size = - round(self.initial_inventory * action / self.trading_steps)
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
        else:
            raise Exception("Unknown Reward Function!")

        self.reward_list.append(reward)

        # Update the trading_environment and get new observation
        self.inventory += order_size
        self.remaining_inventory_list.append(self.inventory)

        done = (self.inventory <= 0)

        info = {'time': self.current_time,
                'shortfall': implementation_shortfall,
                'size': - order_size,
                'price_before_action': self.last_market_price}

        return self.observation_sequence[-self.look_back:], reward, done, info

    def observation(self):
        """
        Take an observation of the current trading_environment

            Returns:
                a vector reflecting the current market condition
        """
        obs = []
        if 'Elapsed Time' in self.ob_dict.keys():
            obs.append((self.current_time - self.initial_time) / self.time_horizon)
        if 'Remaining Inventory' in self.ob_dict.keys():
            obs.append(self.inventory / self.initial_inventory)
        for i in np.arange(1, 11):
            if 'Bid Ask Spread {}'.format(i) in self.ob_dict.keys():
                obs.append(self.OrderBook.getBidAskSpread(i))
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

    def render(self):

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
