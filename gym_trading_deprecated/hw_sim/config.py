import os
from gym_trading_deprecated.hw_sim.OrderBookOracle import OrderBookOracle

"""
config.py is used to add the file path of the historical data while specifying the trading interval
"""

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
orders_file_path = FILE_PATH + '/data/AAPL_2012-06-21_34200000_57600000_message_10.csv'
LOB_file_path = FILE_PATH + '/data/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv'

# mkt_open and mkt_close are in unit "second": 34200 denotes 09:30 and 57600 denotes 16:00
trade_interval = 60
MKT_OPEN = int(34200 / trade_interval)
MKT_CLOSE = int(57600 / trade_interval)
ORDER_BOOK_ORACLE = OrderBookOracle(trade_interval, orders_file_path, LOB_file_path)
