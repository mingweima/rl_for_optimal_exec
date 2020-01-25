import datetime
from collections import deque
import itertools
import threading
import time
import sys
import pickle
from tqdm import tqdm

import pandas as pd

months = ['2018-01-01_2018-01-31',
          '2018-02-01_2018-02-28',
          '2018-03-01_2018-03-31',
          '2018-04-01_2018-04-30',
          '2018-05-01_2018-05-31',
          '2018-06-01_2018-06-30',
          '2018-07-01_2018-07-31',
          '2018-08-01_2018-08-31',
          '2018-09-01_2018-09-30',
          '2018-10-01_2018-10-31',
          '2018-11-01_2018-11-30',
          '2018-12-01_2018-12-31']

for month in months:
    bar = tqdm(range(7))
    bar.set_description('Reading Data -- {}'.format(month))
    path_name = '/nfs/home/mingweim/lob/hsbc/L2_HSBA.L_{}.csv.gz'.format(month)

    raw_data = pd.read_csv(path_name, compression='gzip', error_bad_lines=False)

    bar.update(1)
    bar.set_description('Dropping Columns -- {}'.format(month))

    data = raw_data.drop(['#RIC', 'Domain', 'GMT Offset', 'Type', 'L1-BuyNo', 'L1-SellNo', 'L2-BuyNo', 'L2-SellNo',
                          'L3-BuyNo', 'L3-SellNo', 'L4-BuyNo', 'L4-SellNo', 'L5-BuyNo', 'L5-SellNo',
                          'L6-BuyNo', 'L6-SellNo', 'L7-BuyNo', 'L7-SellNo', 'L8-BuyNo', 'L8-SellNo',
                          'L9-BuyNo', 'L9-SellNo', 'L10-BuyNo', 'L10-SellNo'], axis=1)

    bar.update(1)
    bar.set_description('Rounding to Time Integer -- {}'.format(month))

    data['Date-Time'] = pd.to_datetime(data['Date-Time'],
                                       format='%Y-%m-%dT%H:%M:%S.%fZ').dt.round('{}s'.format(600))

    bar.update(1)
    bar.set_description('Grouping By -- {}'.format(month))

    data = data.groupby(['Date-Time']).first().reset_index()

    bar.update(1)
    bar.set_description('Deleting Weekends -- {}'.format(month))

    data['Day'] = data['Date-Time'].dt.dayofweek
    data = data.drop(data.loc[(data['Day'] == 5) | (data['Day'] == 6)].index)

    bar.update(1)
    bar.set_description('Deleting Auction Periods -- {}'.format(month))

    data['Hour'] = data['Date-Time'].dt.hour
    data['Minute'] = data['Date-Time'].dt.minute
    data = data.drop(
        data.loc[(data['Hour'] < 8) | (data['Hour'] > 16) | ((data['Hour'] == 16) & (data['Minute'] > 0))].index)
    data = data.drop(['Hour', 'Minute', 'Day'], axis=1)

    bar.update(1)
    bar.set_description('Storing Data -- {}'.format(month))

    date = pd.to_datetime(data['Date-Time'].dt.strftime('%Y/%m/%d'))
    unique_date = pd.unique(date)
    for day in unique_date:
        df_train = \
            open('/nfs/home/mingweim/rl_for_optimal_exec/simple_test/data/HSBA/{}_{}.txt'.format(month, day), 'wb')
        pickle.dump(data, df_train)
        df_train.close()

    bar.update(1)
    bar.set_description('Finished Processing Data -- {}'.format(month))
    bar.close()
