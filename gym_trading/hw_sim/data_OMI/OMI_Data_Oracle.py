import os

import numpy as np
import pandas as pd

class OrderBookOracle:
    """
    Oracle for reading historical exchange orders stream
    """
    def __init__(self):

        FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        lob_file_path = FILE_PATH + '/sample.csv'
        lob_df = pd.read_csv(lob_file_path)
        lob_df['Date-Time'] = pd.to_datetime(lob_df['Date-Time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
        self.lob_df = lob_df

    def getHistoricalOrderBook(self, time):
        """
        Output the Limit OrderBook at any historical moment
            Args:
                time: Timestamp
            Returns:
                a list of length two, the first being the bids dictionary, the second the asks dictionary
        """
        LOB = np.array(self.lob_df.loc[self.lob_df['Date-Time'] >= time].head(1))[0]

        bids = []
        for i in range(10):
            price = LOB[6 * i + 5]
            size = LOB[6 * i + 6]
            bids.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'BUY'})
        asks = []
        for i in range(10):
            price = LOB[6 * i + 8]
            size = LOB[6 * i + 9]
            asks.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'SELL'})

        return [bids, asks]
