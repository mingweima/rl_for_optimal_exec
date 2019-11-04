from gym_simulator.envs.OrderBook import OrderBook
from gym_simulator.envs.OrderBookOracle import OrderBookOracle
import pandas as pd
from datetime import datetime
import sys

historical_date = pd.to_datetime('2012-06-21')
mkt_open = historical_date + pd.to_timedelta('34200s')
mkt_close = historical_date + pd.to_timedelta('57600s')
orders_file_path = 'E:/Git/rl_abmnew1/Market_Replay/gym_simulator' \
                   '/envs/GOOG_2012-06-21_34200000_57600000_message_5.csv'
LOB_file_path = 'E:/Git/rl_abmnew1/Market_Replay/gym_simulator' \
                '/envs/GOOG_2012-06-21_34200000_57600000_orderbook_5.csv'

a = OrderBookOracle(mkt_open,mkt_close,orders_file_path,LOB_file_path)
b = a.getHistoricalOrderBook(34201)
a = OrderBook(b)
print(a.asks)


# a = OrderBook()
# order1 = {'PRICE': 150, 'SIZE': 100, 'BUY_SELL_FLAG': 'BUY'}
# order2 = {'PRICE': 140, 'SIZE': 100, 'BUY_SELL_FLAG': 'BUY'}
# order3 = {'PRICE': 130, 'SIZE': 100, 'BUY_SELL_FLAG': 'BUY'}
# order4 = {'PRICE': 120, 'SIZE': 100, 'BUY_SELL_FLAG': 'BUY'}
# a.handleLimitOrder(order1)
# a.handleLimitOrder(order2)
# a.handleLimitOrder(order3)
# a.handleLimitOrder(order4)
# print(a.asks)
# print(a.bids)
# order5 = {'PRICE': 130, 'SIZE': 500, 'BUY_SELL_FLAG': 'SELL'}
# a.handleLimitOrder(order5)
# print(a.asks)
# print(a.bids)
