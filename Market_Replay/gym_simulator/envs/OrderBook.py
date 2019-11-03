import sys
from copy import deepcopy

class OrderBook:
    # An OrderBook requires an owning agent object, which it will use to send messages
    # outbound via the simulator Kernel (notifications of order creation, rejection,
    # cancellation, execution, etc).
    def __init__(self):
        self.bids = []
        self.asks = []
        self.last_trade = None

    def handleLimitOrder(self, order):
        # Matches a limit order or adds it to the order book.  Handles partial matches piecewise,
        # consuming all possible shares at the best price before moving on, without regard to
        # order size "fit" or minimizing number of transactions.

        matching = True
        execution_price = 0
        executed_size = 0

        while matching:
            matched_order = deepcopy(self.executeOrder(order))
            if matched_order:
                # Calculate the execution price.
                execution_price = (execution_price * executed_size + matched_order['PRICE'] *
                                   matched_order['SIZE'])/(executed_size + matched_order['SIZE'])
                executed_size += matched_order['SIZE']
                # Decrement quantity on new order.
                order['SIZE'] -= matched_order['SIZE']
                if order['SIZE'] <= 0:
                    matching = False
            else:
                # No matching order was found, so the new order enters the order book.  Notify the agent.
                self.enterOrder(deepcopy(order))
                matching = False

        return execution_price, executed_size

    def handleMarketOrder(self, action):
        # Input: an integer specifying the size of the market order.
        # Output: weighted average execution price and implementation shortfall.

        if action >= 0:
            lowest_ask_price = self.asks[0]['PRICE']
            order = {'SIZE': action, 'PRICE': sys.maxsize, 'BUY_SELL_FLAG': 'BUY'}
            execution_price, executed_size = self.handleLimitOrder(order)
            implementation_shortfall = (execution_price - lowest_ask_price) * executed_size
        else:
            highest_bid_price = self.bids[0]['PRICE']
            order =  {'SIZE': -action, 'PRICE': 0, 'BUY_SELL_FLAG': 'SELL'}
            execution_price, executed_size = self.handleLimitOrder(order)
            implementation_shortfall = (highest_bid_price - execution_price) * executed_size

        if executed_size != abs(action):
            raise ValueError("Size of the Market Order cannot exceed the size of LOB! ")

        return execution_price, implementation_shortfall


    def executeOrder(self, order):
        # Finds a single best match for this order, without regard for quantity.
        # Returns the matched order or None if no match found.  DOES remove,
        # or decrement quantity from, the matched order from the order book
        # (i.e. executes at least a partial trade, if possible).

        # Track which (if any) existing order was matched with the current order.
        if order['BUY_SELL_FLAG'] == 'BUY':
            book = self.asks
        else:
            book = self.bids
        # First, examine the correct side of the order book for a match.
        if not book:
            # No orders on this side.
            return None
        elif not self.isMatch(order, book[0][0]):
            # There were orders on the right side, but the prices do not overlap.
            # Or: bid could not match with best ask, or vice versa.
            # Or: bid offer is below the lowest asking price, or vice versa.
            return None
        else:
            # There are orders on the right side, and the new order's price does fall
            # somewhere within them.  We can/will only match against the oldest order
            # among those with the best price.  (i.e. best price, then FIFO)

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
        # Returns True if order 'o' can be matched against input 'order'.
        if order['BUY_SELL_FLAG'] == o['BUY_SELL_FLAG']:
            return False
        if order['BUY_SELL_FLAG'] == 'BUY' and (order['PRICE'] >= o['PRICE']):
            return True
        if order['BUY_SELL_FLAG'] == 'SELL' and (order['PRICE'] <= o['PRICE']):
            return True
        return False

    def enterOrder(self, order):
        # Enters a limit order into the OrderBook in the appropriate location.
        # This does not test for matching/executing orders -- this function
        # should only be called after a failed match/execution attempt.
        if order['BUY_SELL_FLAG'] == 'BUY':
            book = self.bids
        else:
            book = self.asks
        if not book:
            # There were no orders on this side of the book.
            book.append([order])
        elif not self.isBetterPrice(order, book[-1][0]) and not self.isEqualPrice(order, book[-1][0]):
            # There were orders on this side, but this order is worse than all of them.
            # (New lowest bid or highest ask.)
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

    # Get the inside bid price(s) and share volume available at each price, to a limit
    # of "depth".   Returns a list of [price, total shares]:
    def getInsideBids(self, depth=sys.maxsize):
        book = []
        for i in range(min(depth, len(self.bids))):
            qty = 0
            price = self.bids[i][0]['PRICE']
            for o in self.bids[i]:
                qty += o['SIZE']
            book.append([price, qty])
        return book

    # As above, except for ask price(s).
    def getInsideAsks(self, depth=sys.maxsize):
        book = []
        for i in range(min(depth, len(self.asks))):
            qty = 0
            price = self.asks[i][0]['PRICE']
            for o in self.asks[i]:
                qty += o['SIZE']
            book.append([price, qty])
        return book

    def isBetterPrice(self, order, o):
        # Returns True if order has a 'better' price than o.  (That is, a higher bid
        # or a lower ask.)  Must be same order type.
        if order['BUY_SELL_FLAG'] == 'BUY' and (order['PRICE'] > o['PRICE']):
            return True
        if order['BUY_SELL_FLAG'] == 'SELL' and (order['PRICE'] < o.limit_price):
            return True
        return False

    def isEqualPrice(self, order, o):
        return order['PRICE'] == o['PRICE']

    def isSameOrder(self, order, new_order):
        return order['ORDER_ID'] == new_order['ORDER_ID']

    def getMidPrice(self):
        return (self.bids[0]['PRICE'] + self.asks[0]['PRICE'])/2

    def getBidAskSpread(self):
        return self.asks[0]['PRICE'] - self.bids[0]['PRICE']