import os
import itertools
import threading
import time
import sys


import numpy as np
import pandas as pd

class OrderBookOracle:
    """
    Oracle for reading historical exchange orders stream
    """
    def __init__(self, data_args, initial_time, trading_interval, time_horizon):

        done = False

        def animate():
            interval = 0.5
            decay = 0.98
            for c in itertools.cycle(['|', '/', '-', '\\']):
                if done:
                    break
                sys.stdout.write('\rLoading OMI Data ' + c)
                sys.stdout.flush()
                time.sleep(interval)
                if interval >= 0.03:
                    interval *= decay
            sys.stdout.write('\rDone!     ')

        t = threading.Thread(target=animate)
        t.start()

        FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        lob_file_path = FILE_PATH + '/{}'.format(data_args)
        lob_df = pd.read_csv(lob_file_path)
        lob_df = lob_df.drop(['#RIC', 'Domain', 'GMT Offset', 'Type', 'L1-BuyNo', 'L1-SellNo', 'L2-BuyNo', 'L2-SellNo',
                              'L3-BuyNo', 'L3-SellNo', 'L4-BuyNo', 'L4-SellNo', 'L5-BuyNo', 'L5-SellNo',
                              'L6-BuyNo', 'L6-SellNo', 'L7-BuyNo',  'L7-SellNo', 'L8-BuyNo', 'L8-SellNo',
                              'L9-BuyNo', 'L9-SellNo', 'L10-BuyNo', 'L10-SellNo'], axis=1)
        lob_df['Date-Time'] = pd.to_datetime(lob_df['Date-Time'],
                                             format='%Y-%m-%dT%H:%M:%S.%fZ').dt.round('{}s'.format(trading_interval))
        lob_df = lob_df.groupby(['Date-Time']).first().reset_index()
        lob_df = lob_df.loc[(lob_df['Date-Time'] >= initial_time) &
                            (lob_df['Date-Time'] <= initial_time + time_horizon)]
        self.lob_df = lob_df

        done = True

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
            price = LOB[4 * i + 1]
            size = LOB[4 * i + 2]
            bids.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'BUY'})
        asks = []
        for i in range(10):
            price = LOB[4 * i + 3]
            size = LOB[4 * i + 4]
            asks.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': price, 'SIZE': size, 'BUY_SELL_FLAG': 'SELL'})

        return [bids, asks]
