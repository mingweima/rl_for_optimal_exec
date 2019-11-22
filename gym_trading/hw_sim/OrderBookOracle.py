import pandas as pd
import numpy as np


class OrderBookOracle:
#     Oracle for reading historical exchange orders stream
    def __init__(self, trade_interval, orders_file_path, LOB_file_path):
        self.trade_interval = trade_interval
        self.orders_file_path = orders_file_path
        self.LOB_file_path = LOB_file_path
        self.orders_list, self.LOB_df = self.processData()

    def getHistoricalOrderBook(self, time):
        # Output the Limit OrderBook at any historical moment
        i = 1
        while True:
            if sum(self.LOB_df['TIME'] == time):
                LOB = np.array(self.LOB_df.loc[self.LOB_df['TIME'] == time - i].tail(1))[0]
                break
            else:
                i += 1
        # Generating the Ask Order Book
        asks = []
        for i in range(10):
            price = LOB[4 * i]
            size = LOB[4 * i + 1]
            asks.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'SELL'})
        # Generating the Bid Order Book
        bids = []
        for i in range(10):
            price = LOB[4 * i + 2]
            size = LOB[4 * i + 3]
            bids.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'BUY'})
        asks.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': 6000000, 'SIZE': 100000, 'BUY_SELL_FLAG': 'SELL'})
        bids.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': 5700000, 'SIZE': 100000, 'BUY_SELL_FLAG': 'BUY'})
        return [bids, asks]

    def processData(self):
        columns = ['TIME', 'TYPE', 'ORDER_ID', 'SIZE', 'PRICE', 'BUY_SELL_FLAG']
        orders_df = pd.read_csv(self.orders_file_path, header=None, names=columns)
        orders_df['BUY_SELL_FLAG'] = orders_df['BUY_SELL_FLAG'].replace({-1: 'SELL', 1: 'BUY'})
        orders_df['TIME'] = (orders_df['TIME'] / self.trade_interval).astype(int)
        orders_df['PRICE'] = orders_df['PRICE']
        orders_list = orders_df.to_dict('records')

        LOB_df = pd.read_csv(self.LOB_file_path, header=None)
        LOB_df['TIME'] = orders_df['TIME']

        # orders_list is a list of dictionaries listing all the orders
        # LOB_df is a DataFrame of the configuration of the LOB
        return orders_list, LOB_df
