import numpy as np

from trading_environment.orderbook import OrderBook

STEPS = 24

class Simulator:
    """
    The gym trading_environment for optimal execution: managing a limit order book by taking in historical
    orders and reacting to the actions of the agent.
    """

    def __init__(self, data_dict, date_dict, ac_dict, ob_dict, initial_shares, look_back, price_smooth):
        self.data_dict = data_dict
        self.date_dict = date_dict
        self.price_smooth = price_smooth

        self.hothead = 'False'
        self.trading_steps = STEPS
        self.initial_inventory = initial_shares
        self.look_back = look_back

        # Initialize the action space
        self.ac_dict = ac_dict

        # Initialize the observation space
        self.ob_dict = {k: v for k, v in ob_dict.items() if v}

        # Initialize the reward function
        self.reward_function = 'implementation_shortfall'

    def reset(self, month, day, session):
        """
        Reset the trading_environment before the start of the experiment or after finishing one trial.

            Returns:
                obs (nd array): the current observation
        """

        # Initialize the OrderBook
        self.data = self.data_dict[month][day][session]

        # mid_price_list = []
        # volume_list = []
        #
        # # 24 steps of normalization (2 hours)
        # for interval in range(24):
        #     LOB = np.array(self.data.loc[interval])
        #     mid_price = (LOB[1] + LOB[3]) / 2
        #     if mid_price:
        #         mid_price_list.append(mid_price)
        #     volume = (sum(LOB[4 * j + 2] for j in range(10)) + sum(LOB[4 * j + 4] for j in range(10))) / 20
        #     if volume:
        #         volume_list.append(volume)
        #
        # self.price_mean, self.price_std, self.volume_mean, self.volume_std = \
        #     np.average(mid_price_list), np.std(mid_price_list), np.average(volume_list), np.std(volume_list)

        self.inventory = self.initial_inventory

        self.initial_loc = 24
        self.current_loc = 24
        self.OrderBook = OrderBook(self.get_historical_order())
        # self.arrival_price = self.OrderBook.getMidPrice()
        self.arrival_price = self.OrderBook.get_hothead_vwap(-self.initial_inventory)

        # Initialize the observation sequence
        self.current_loc = 0
        self.observation_sequence = []
        self.bid_price_sequence = [[]] * self.price_smooth
        self.ask_price_sequence = [[]] * self.price_smooth
        self.OrderBook = OrderBook(self.get_historical_order())
        while self.current_loc <= 24:
            self.OrderBook.update(self.get_historical_order())
            for level in np.arange(1, 11):
                if 'Bid Price {}'.format(level) in self.ob_dict.keys():
                    self.bid_price_sequence[level - 1].append(self.OrderBook.getBidsPrice(level))
                if 'Ask Price {}'.format(level) in self.ob_dict.keys():
                    self.ask_price_sequence[level - 1].append(self.OrderBook.getAsksPrice(level))
            self.observation_sequence.append(self.observation())
            self.current_loc += 1

        self.current_loc = 24
        self.OrderBook = OrderBook(self.get_historical_order())
        return self.observation_sequence[-self.look_back:]

    def get_historical_order(self):
        LOB = np.array(self.data.loc[self.current_loc])

        bids = []
        for i in range(10):
            price = LOB[4 * i + 1]
            size = LOB[4 * i + 2]
            bids.append({'TYPE': 0, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'BUY'})
        asks = []
        for i in range(10):
            price = LOB[4 * i + 3]
            size = LOB[4 * i + 4]
            asks.append({'TYPE': 0, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'SELL'})

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
        price = self.OrderBook.getMidPrice()

        # Place the agent's order to the limit order book
        if action == -1:
            order_size = - self.inventory
        elif self.current_loc >= 47:
            order_size = - self.inventory
        elif action == -2:
            order_size = - round(self.initial_inventory / self.trading_steps)
        else:
            action = self.ac_dict[action]
            order_size = - round(self.initial_inventory * action / self.trading_steps)
        if self.inventory + order_size < 0:
            order_size = - self.inventory

        if order_size != 0:
            vwap, _ = self.OrderBook.handleMarketOrder(order_size)
        else:
            vwap = 0

        implementation_shortfall = - 1000 * (order_size / self.initial_inventory) * \
                                   (vwap - self.arrival_price) / self.arrival_price

        # Calculate the reward
        if self.reward_function == 'implementation_shortfall':
            reward = implementation_shortfall
        else:
            raise Exception("Unknown Reward Function!")

        # Update the trading_environment and get new observation
        self.inventory += order_size

        done = (self.inventory <= 0)

        info = {'step': self.current_loc - 23,
                'shortfall': implementation_shortfall,
                'size': - order_size,
                'price': price}

        # Update the time
        self.current_loc += 1

        # Update the LOB. (for OMI data)
        self.OrderBook.update(self.get_historical_order())
        for level in np.arange(1, 11):
            if 'Bid Price {}'.format(level) in self.ob_dict.keys():
                self.bid_price_sequence[level - 1].append(self.OrderBook.getBidsPrice(level))
            if 'Ask Price {}'.format(level) in self.ob_dict.keys():
                self.ask_price_sequence[level - 1].append(self.OrderBook.getAsksPrice(level))

        # Take an observation of the current state
        obs = self.observation()
        self.observation_sequence.append(obs)

        return self.observation_sequence[-self.look_back:], reward, done, info

    def observation(self):
        """
        Take an observation of the current trading_environment

            Returns:
                a vector reflecting the current market condition
        """
        obs = []
        if 'Elapsed Time' in self.ob_dict.keys():
            obs.append((self.current_loc - self.initial_loc) / self.trading_steps)
        if 'Remaining Inventory' in self.ob_dict.keys():
            obs.append(self.inventory / self.initial_inventory)

        for i in np.arange(1, 11):
            if 'Bid L{} VWAP'.format(i) in self.ob_dict.keys():
                obs.append((self.OrderBook.bids_vwap(i) - self.price_mean) / self.price_std)
            if 'Ask L{} VWAP'.format(i) in self.ob_dict.keys():
                obs.append((self.OrderBook.asks_vwap(i) - self.price_mean) / self.price_std)
            if 'Bid L{} Volume'.format(i) in self.ob_dict.keys():
                obs.append((self.OrderBook.sum_bids_qty(i) - self.volume_mean * i) / (self.volume_std * np.sqrt(i)))
            if 'Ask L{} Volume'.format(i) in self.ob_dict.keys():
                obs.append((self.OrderBook.sum_asks_qty(i) - self.volume_mean * i) / (self.volume_std * np.sqrt(i)))

        for i in np.arange(1, 11):
            if 'Bid Ask Spread {}'.format(i) in self.ob_dict.keys():
                obs.append(self.OrderBook.getBidAskSpread(i))

        for i in np.arange(1, 11):
            if 'Bid Ask Spread {}'.format(i) in self.ob_dict.keys():
                obs.append(self.OrderBook.getBidAskSpread(i))
            if 'Bid Price {}'.format(i) in self.ob_dict.keys():
                # obs.append((self.OrderBook.getBidsPrice(i) - self.price_mean) / self.price_std)
                print(self.bid_price_sequence)
                price = np.average(self.bid_price_sequence[i - 1][-self.price_smooth:])
                bp = 100 * (price - self.arrival_price) / self.arrival_price
                obs.append(bp)
            if 'Ask Price {}'.format(i) in self.ob_dict.keys():
                price = np.average(self.ask_price_sequence[i - 1][-self.price_smooth:])
                ap = 100 * (price - self.arrival_price) / self.arrival_price
                obs.append(ap)
                # obs.append((self.OrderBook.getAsksPrice(i) - self.price_mean) / self.price_std)
            if 'Bid Volume {}'.format(i) in self.ob_dict.keys():
                # obs.append((self.OrderBook.getBidsQuantity(i) - self.volume_mean) / self.volume_std)
                bv = (self.OrderBook.getBidsQuantity(i) - (self.initial_inventory / 24)) / (self.initial_inventory / 24)
                obs.append(bv)
            if 'Ask Volume {}'.format(i) in self.ob_dict.keys():
                # obs.append((self.OrderBook.getAsksQuantity(i) - self.volume_mean) / self.volume_std)
                av = (self.OrderBook.getAsksQuantity(i) - (self.initial_inventory / 24)) / (self.initial_inventory / 24)
                obs.append(av)

        return np.asarray(obs)

    def render(self):
        pass
