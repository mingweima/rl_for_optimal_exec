import sys
from copy import deepcopy


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
        self.base_price = initial_orders[0][-1]['PRICE']
        self.handleLimitOrder({'TYPE': 0, 'ORDER_ID': -1, 'PRICE': self.base_price, 'SIZE': 1e8, 'BUY_SELL_FLAG': 'BUY'})

    def update(self, historical_orders):
        self.bids = []
        self.asks = []
        for bid_order in historical_orders[0]:
            self.handleLimitOrder(bid_order)
        for ask_order in historical_orders[1]:
            self.handleLimitOrder(ask_order)
        self.base_price = historical_orders[0][-1]['PRICE']
        self.handleLimitOrder({'TYPE': 0, 'ORDER_ID': -1, 'PRICE': self.base_price, 'SIZE': 1e8, 'BUY_SELL_FLAG': 'BUY'})
        print(self.bids)

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
        print(action)
        print(executed_size)
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

    def sum_asks_qty(self, level):
        qty = 0
        for level in range(1, level + 1):
            for o in self.asks[level - 1]:
                qty += o['SIZE']
        return qty

    def sum_bids_qty(self, level):
        qty = 0
        for level in range(1, level + 1):
            for o in self.bids[level - 1]:
                qty += o['SIZE']
        return qty

    def bids_vwap(self, level):
        p = 0
        for level in range(1, level + 1):
            for o in self.bids[level - 1]:
                p += o['SIZE'] * o['PRICE']
        return p / self.sum_bids_qty(level)

    def asks_vwap(self, level):
        p = 0
        for level in range(1, level + 1):
            for o in self.asks[level - 1]:
                p += o['SIZE'] * o['PRICE']
        return p / self.sum_asks_qty(level)

    def getTotalBidsQuantity(self):
        qty = 0
        for level in self.bids:
            for o in level:
                qty += o['SIZE']
        return qty

    def getAsksPrice(self, level):
        return self.asks[level - 1][0]['PRICE']

    def get_base_ask_price(self):
        return self.base_price

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

    def get_hothead_vwap(self, shares):
        vwap, _ = self.handleMarketOrder(shares)
        return vwap

    def getBidAskSpread(self, level):
        """
        Returns the current bid-ask spread.
        """
        try:
            return self.asks[level - 1][0]['PRICE'] - self.bids[level - 1][0]['PRICE']
        except:
            raise Exception("For this specific moment, the LOB does not have enough levels! \n"
                            "Please reset the observation space to involve only lower levels of LOB.")
