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
    def __init__(self, data_args, trading_interval):

        done = False

        def animate():
            for c in itertools.cycle(['|', '/', '-', '\\']):
                if done:
                    break
                sys.stdout.write('\rLoading OMI Data ' + c)
                sys.stdout.flush()
                time.sleep(0.1)

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
        lob_df['Day'] = lob_df['Date-Time'].dt.dayofweek
        lob_df = lob_df.drop(lob_df.loc[(lob_df['Day'] == 5) | (lob_df['Day'] == 6)].index)
        self.lob_df = lob_df

        date = pd.to_datetime(self.lob_df['Date-Time'].dt.strftime('%Y/%m/%d'))
        self.unique_date = pd.unique(date)
        done = True

    def get_past_price_volume(self, current_date, days):
        """
        This function is used for normalization.
            Args:
                current_date (Timestamp): the current date
                days (int): number of days to trace back
            Returns:
                the mean and standard deviation of price and volume
        """
        mid_price_list = np.zeros(days)
        volume_list = np.zeros(days)
        idx = list(self.unique_date).index(current_date)
        for i in range(days):
            LOB = np.array(self.lob_df.loc[self.lob_df['Date-Time'] >=
                                           self.unique_date[idx - i - 1] + pd.Timedelta('9hours')].head(1))[0]
            mid_price = (LOB[1] + LOB[3]) / 2
            mid_price_list[i] = mid_price
            volume = sum(LOB[4 * j + 2] for j in range(10)) + sum(LOB[4 * j + 4] for j in range(10))
            volume_list[i] = volume
        return np.average(mid_price_list), np.std(mid_price_list), np.average(volume_list), np.std(volume_list)

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

        # Add a very large order with bad price at the bottom of the order book to avoid order depletion
        asks.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': 10000, 'SIZE': 100000, 'BUY_SELL_FLAG': 'SELL'})
        bids.append({'TIME': time, 'TYPE': 1, 'ORDER_ID': -1, 'PRICE': 0, 'SIZE': 100000, 'BUY_SELL_FLAG': 'BUY'})

        return [bids, asks]
