import random
import sys
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class OrderBook:
    """
    An OrderBook maintains a bid book and an ask book.
    The OrderBook handles limit and market orders either by executing it or adding it into the bid/ask book.
    The OrderBook also conduct easy calculation including mid_price and spread.

        Attributes:
            bids (list): a list of dictionaries of bid orders
            asks (list): a list of dictionaries of ask orders
    """
    def __init__(self, initial_orders):
        self.bids = []
        self.asks = []

        # Takes in initial orders to initialize the OrderBook before any operations.
        for bid_order in initial_orders[0]:
            self.handleLimitOrder(bid_order)
        for ask_order in initial_orders[1]:
            self.handleLimitOrder(ask_order)

    def update(self, historical_orders):
        self.bids = []
        self.asks = []

        for bid_order in historical_orders[0]:
            self.handleLimitOrder(bid_order)
        for ask_order in historical_orders[1]:
            self.handleLimitOrder(ask_order)
        self.handleLimitOrder({'TYPE': 0, 'ORDER_ID': -1, 'PRICE': 0, 'SIZE': 10000, 'BUY_SELL_FLAG': 'BUY'})

    def handleLimitOrder(self, input_order):
        """
        Matches a limit order or adds it to the order book.
        Returns execution price and executed size, if the order is completed added to the order book without
            any matching, both execution price and executed size are set to zero.

            Args:
                input_order (dictionary): the order to handle
            Returns:
                execution_price (float64)
                execution_size (float64)
        """

        execution_price = 0.0
        executed_size = 0

        order = deepcopy(input_order)
        if order['TYPE'] == 1:
            # Order type 1 corresponds to new limit order.
            # Repeatedly match the order with the order book
            matching = True
            while matching:
                matched_order = deepcopy(self.executeOrder(order))
                if matched_order:
                    # Update the execution price and executed price
                    if executed_size + matched_order['SIZE'] == 0:
                        execution_price = 0
                    else:
                        execution_price = (execution_price * executed_size + matched_order['PRICE'] *
                                        matched_order['SIZE'])/(executed_size + matched_order['SIZE'])
                    executed_size += matched_order['SIZE']

                    # Decrement quantity on the order.
                    order['SIZE'] -= matched_order['SIZE']
                    if order['SIZE'] <= 0:
                        matching = False
                else:
                    # No matching order was found, so the new order enters the order book.
                    self.enterOrder(deepcopy(order))
                    matching = False

        if order['TYPE'] == 0:
            self.enterOrder(deepcopy(order))

        return execution_price, executed_size

    def handleMarketOrder(self, action):
        """
        Handle an market order.
            Args:
                action (int32):
                    an integer specifying the size of the market order (Buy: positive; Sell: Negative).
            Returns:
                execution_price (float64): weighted average execution price
                implementation shortfall (float64): the order-wise implementation shortfall.

        """
        if action >= 0:
            lowest_ask_price = self.asks[0][0]['PRICE']
            order = {'TYPE': 1, 'SIZE': action, 'ORDER_ID': -1, 'PRICE': sys.maxsize, 'BUY_SELL_FLAG': 'BUY'}

            # Handles the corresponding limit order.
            execution_price, executed_size = self.handleLimitOrder(order)
            implementation_shortfall = (execution_price - lowest_ask_price) * executed_size

        else:
            highest_bid_price = self.bids[0][0]['PRICE']
            order =  {'TYPE': 1, 'SIZE': -action, 'ORDER_ID': -1, 'PRICE': 0, 'BUY_SELL_FLAG': 'SELL'}

            # Handles the corresponding limit order.
            execution_price, executed_size = self.handleLimitOrder(order)
            implementation_shortfall = (highest_bid_price - execution_price) * executed_size

        # Size of the order cannot exceed the size of LOB.
        if executed_size != abs(action):
            raise ValueError("Size of the Market Order cannot exceed the size of LOB! ")

        return execution_price, implementation_shortfall


    def executeOrder(self, order):
        """
        Finds a single best match for this order, without regard for quantity.
        Returns the matched order or None if no match found.
        Remove or decrement quantity from the matched order from the order book
        """

        # Which order book (bid or ask) should we look at?
        if order['BUY_SELL_FLAG'] == 'BUY':
            book = self.asks
        else:
            book = self.bids

        if not book:
            # No orders in the book.
            return None
        elif not self.isMatch(order, book[0][0]):
            # There were orders on the right side, but the prices do not match.
            return None
        else:
            # Note that book[i] is a LIST of all orders (oldest at index book[i][0]) at the same price.
            # The matched order might be only partially filled. (i.e. new order is smaller)
            if order['SIZE'] >= book[0][0]['SIZE']:
                # Consumed entire matched order.
                matched_order = book[0].pop(0)

                # If the matched price now has no orders, remove it completely.
                if not book[0]:
                    del book[0]
            else:
                # Consumed only part of matched order.
                matched_order = deepcopy(book[0][0])
                matched_order['SIZE'] = order['SIZE']
                book[0][0]['SIZE'] -= matched_order['SIZE']

            # Return (only the executed portion of) the matched order.
            return matched_order

    def isMatch(self, order, o):
        """
        Returns True if order 'o' can be matched against input 'order'.
        """

        if order['BUY_SELL_FLAG'] == o['BUY_SELL_FLAG']:
            return False
        elif order['BUY_SELL_FLAG'] == 'BUY' and (order['PRICE'] >= o['PRICE']):
            return True
        elif order['BUY_SELL_FLAG'] == 'SELL' and (order['PRICE'] <= o['PRICE']):
            return True
        else: return False

    def enterOrder(self, order):
        """
        Enters a limit order into the OrderBook in the appropriate location.
        """

        if order['BUY_SELL_FLAG'] == 'BUY':
            book = self.bids
        else:
            book = self.asks
        if not book:
            # There were no orders on this side of the book.
            book.append([order])
        elif not self.isBetterPrice(order, book[-1][0]) and not self.isEqualPrice(order, book[-1][0]):
            # There were orders on this side, but this order is worse than all of them.
            book.append([order])
        else:
            # There are orders on this side.  Insert this order in the correct position in the list.
            # Note that o is a LIST of all orders (oldest at index 0) at this same price.
            for i, o in enumerate(book):
                if self.isBetterPrice(order, o[0]):
                    book.insert(i, [order])
                    break
                elif self.isEqualPrice(order, o[0]):
                    book[i].append(order)
                    break

    def getInsideBids(self, depth=sys.maxsize):
        """
        Get the inside bid price(s) and share volume available at each price, to a limit
        of "depth".   Returns a list of [price, total shares]
        """

        book = []
        for i in range(min(depth, len(self.bids))):
            qty = 0
            price = self.bids[i][0]['PRICE']
            for o in self.bids[i]:
                qty += o['SIZE']
            book.append([price, qty])
        return book

    def getInsideAsks(self, depth=sys.maxsize):
        """
        Get the inside ask price(s) and share volume available at each price, to a limit
        of "depth".   Returns a list of [price, total shares]
        """

        book = []
        for i in range(min(depth, len(self.asks))):
            qty = 0
            price = self.asks[i][0]['PRICE']
            for o in self.asks[i]:
                qty += o['SIZE']
            book.append([price, qty])
        return book

    def getAsksQuantity(self, level):
        qty = 0
        for o in self.asks[level - 1]:
            qty += o['SIZE']
        return qty

    def getBidsQuantity(self, level):
        qty = 0
        for o in self.bids[level - 1]:
            qty += o['SIZE']
        return qty

    def getTotalBidsQuantity(self):
        qty = 0
        for level in self.bids:
            for o in level:
                qty += o['SIZE']
        return qty

    def getAsksPrice(self, level):
        return self.asks[level - 1][0]['PRICE']

    def getBidsPrice(self, level):
        return self.bids[level - 1][0]['PRICE']

    def isBetterPrice(self, order, o):
        """
        Returns True if order has a 'better' price than o.  (That is, a higher bid
        or a lower ask.)  Must be same order type.
        """

        if order['BUY_SELL_FLAG'] == 'BUY' and (order['PRICE'] > o['PRICE']):
            return True
        elif order['BUY_SELL_FLAG'] == 'SELL' and (order['PRICE'] < o['PRICE']):
            return True
        else:
            return False

    def isEqualPrice(self, order, o):
        return order['PRICE'] == o['PRICE']

    def isSameOrder(self, order, new_order):
        return order['ORDER_ID'] == new_order['ORDER_ID']

    def getMidPrice(self):
        """
        Returns the current mid-price.
        """

        if self.asks and self.bids:
            return (self.bids[0][0]['PRICE'] + self.asks[0][0]['PRICE'])/2
        else:
            return -1

    def getBidAskSpread(self, level):
        """
        Returns the current bid-ask spread.
        """
        return self.asks[level - 1][0]['PRICE'] - self.bids[level - 1][0]['PRICE']




class Simulator:
    """
    The gym environment for optimal execution: managing a limit order book by taking in historical
    orders and reacting to the actions of the agent.
    """

    def __init__(self, data_dict, date_dict, ac_dict):
        self.data_dict = data_dict
        self.date_dict = date_dict

        self.hothead = 'False'
        self.trading_interval = 600
        self.time_horizon = pd.Timedelta(seconds=18000)
        self.trading_steps = int(self.time_horizon.seconds / self.trading_interval)
        self.initial_inventory = 3000
        self.look_back = 12

        # Initialize the action space
        self.ac_dict = ac_dict
        self.ac_type = 'prop of linear'

        # Initialize the observation space
        ob_dict = {
            'Elapsed Time': True,
            'Remaining Inventory': True,
            'Bid Ask Spread 1': True,
            'Bid Ask Spread 2': True,
            'Bid Ask Spread 3': True,
            'Bid Ask Spread 4': True,
            'Bid Ask Spread 5': True,
            # 'Bid Ask Spread 6': True,
            # 'Bid Ask Spread 7': True,
            # 'Bid Ask Spread 8': True,
            # 'Bid Ask Spread 9': True,
            # 'Bid Ask Spread 10': True,
            'Bid Price 1': True,
            'Bid Price 2': True,
            'Bid Price 3': True,
            'Bid Price 4': True,
            'Bid Price 5': True,
            # 'Bid Price 6': True,
            # 'Bid Price 7': True,
            # 'Bid Price 8': True,
            # 'Bid Price 9': True,
            # 'Bid Price 10': True,
            'Bid Volume 1': True,
            'Bid Volume 2': True,
            'Bid Volume 3': True,
            'Bid Volume 4': True,
            'Bid Volume 5': True,
            # 'Bid Volume 6': True,
            # 'Bid Volume 7': True,
            # 'Bid Volume 8': True,
            # 'Bid Volume 9': True,
            # 'Bid Volume 10': True,
            'Ask Price 1': True,
            'Ask Price 2': True,
            'Ask Price 3': True,
            'Ask Price 4': True,
            'Ask Price 5': True,
            # 'Ask Price 6': True,
            # 'Ask Price 7': True,
            # 'Ask Price 8': True,
            # 'Ask Price 9': True,
            # 'Ask Price 10': True,
            'Ask Volume 1': True,
            'Ask Volume 2': True,
            'Ask Volume 3': True,
            'Ask Volume 4': True,
            'Ask Volume 5': True,
            # 'Ask Volume 6': True,
            # 'Ask Volume 7': True,
            # 'Ask Volume 8': True,
            # 'Ask Volume 9': True,
            # 'Ask Volume 10': True,
        }
        self.ob_dict = {k: v for k, v in ob_dict.items() if v}

        # Initialize the reward function
        self.reward_function = 'implementation_shortfall'

        # Initialize the Oracle by imputing historical data files.
        # date = pd.to_datetime(self.data['Date-Time'].dt.strftime('%Y/%m/%d'))
        # self.unique_date = pd.unique(date)


    def reset(self, month, day):
        """
        Reset the environment before the start of the experiment or after finishing one trial.

            Returns:
                obs (nd array): the current observation
        """

        # Initialize the OrderBook
        # initial_date = random.choice(self.unique_date[-10:])
        # initial_date = self.unique_date[num_days]
        self.data = self.data_dict[month]
        initial_date = day

        print(self.data)
        print(day + pd.Timedelta('11hours'))

        mid_price_list = []
        volume_list = []

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
                    print('Cannot find LOB for ', day + pd.Timedelta('{}hours'.format(11 - hour))
                          + pd.Timedelta('{}hours'.format(interval)))
                    print('Use instead LOB for ', day + pd.Timedelta('{}hours'.format(11 - hour - 1))
                          + pd.Timedelta('{}hours'.format(interval)))
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
            LOB = np.array(self.data.loc[self.data['Date-Time'] >= self.current_time - pd.Timedelta('{}hours'.format(hour))].head(1))
            try:
                LOB = LOB[0]
                break
            except:
                print('Cannot find LOB for ', self.current_time - pd.Timedelta('{}hours'.format(hour)))
                print('Use instead LOB for ', self.current_time - pd.Timedelta('{}hours'.format(hour + 1)))
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

        # Update the environment and get new observation
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
