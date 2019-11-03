from gym_simulator.envs.OrderBook import OrderBook
from gym_simulator.envs.OrderBookOracle import OrderBookOracle
import pandas as pd
from datetime import datetime
import sys

historical_date = pd.to_datetime('2019-06-03')
mkt_open = historical_date + pd.to_timedelta('09:30:00')
mkt_close = historical_date + pd.to_timedelta('16:00:00')
orders_file_path = 'E:/Git/rl_abmnew1/Market_Replay/gym_simulator/envs/sample_orders_file.csv'
a=2
print(-a)
# print(mkt_open + pd.to_timedelta('{}ms'.format(1)))
#
# a = OrderBookOracle(mkt_open,mkt_close,orders_file_path)
# print(a.orders_list)


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
