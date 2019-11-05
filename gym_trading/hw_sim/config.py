import os
from gym_trading.hw_sim.OrderBookOracle import OrderBookOracle


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
orders_file_path = FILE_PATH + '/AAPL_2012-06-21_34200000_57600000_message_10.csv'
LOB_file_path = FILE_PATH + '/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv'

# mkt_open and mkt_close are in unit "second": 34200 denotes 09:30 and 57600 denotes 16:00
MKT_OPEN = 34200
MKT_CLOSE = 57600

ORDER_BOOK_ORACLE = OrderBookOracle(MKT_OPEN, MKT_CLOSE, orders_file_path, LOB_file_path)