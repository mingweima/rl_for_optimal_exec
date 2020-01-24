import datetime
from collections import deque
import itertools
import threading
import time
import sys
import pickle

import pandas as pd

done = False

def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rProcessing OMI Data ' + c)
        sys.stdout.flush()
        time.sleep(0.1)

t = threading.Thread(target=animate)
t.start()

# train_paths = [f'/nfs/home/mingweim/lob/hsbc/L2_HSBA.L_{month}.csv.gz'
#                 for month in
#                ['2018-01-01_2018-01-31', '2018-02-01_2018-02-28', '2018-03-01_2018-03-31', '2018-04-01_2018-04-30',
#                 '2018-05-01_2018-05-31', '2018-06-01_2018-06-30', '2018-07-01_2018-07-31', '2018-08-01_2018-08-31']]

train_paths = [f'/nfs/home/mingweim/lob/hsbc/L2_HSBA.L_2018-02-01_2018-02-28.csv.gz']
test_paths = [f'/nfs/home/mingweim/lob/hsbc/L2_HSBA.L_2018-01-01_2018-01-31.csv.gz']

# test_paths =  [f'/nfs/home/mingweim/lob/hsbc/L2_HSBA.L_{month}.csv.gz'
#                 for month in
#                ['2018-09-01_2018-09-30', '2018-10-01_2018-10-31', '2018-11-01_2018-11-30', '2018-12-01_2018-12-31']]


train_raw_data_list = [pd.read_csv(path_name, compression='gzip', error_bad_lines=False) for path_name in train_paths]
test_raw_data_list = [pd.read_csv(path_name, compression='gzip', error_bad_lines=False) for path_name in test_paths]

train_data_list = []
test_data_list = []

for raw_data in train_raw_data_list:
    data = raw_data.drop(['#RIC', 'Domain', 'GMT Offset', 'Type', 'L1-BuyNo', 'L1-SellNo', 'L2-BuyNo', 'L2-SellNo',
                          'L3-BuyNo', 'L3-SellNo', 'L4-BuyNo', 'L4-SellNo', 'L5-BuyNo', 'L5-SellNo',
                          'L6-BuyNo', 'L6-SellNo', 'L7-BuyNo', 'L7-SellNo', 'L8-BuyNo', 'L8-SellNo',
                          'L9-BuyNo', 'L9-SellNo', 'L10-BuyNo', 'L10-SellNo'], axis=1)
    data['Date-Time'] = pd.to_datetime(data['Date-Time'],
                                       format='%Y-%m-%dT%H:%M:%S.%fZ').dt.round('{}s'.format(600))
    data = data.groupby(['Date-Time']).first().reset_index()
    data['Day'] = data['Date-Time'].dt.dayofweek
    data = data.drop(data.loc[(data['Day'] == 5) | (data['Day'] == 6)].index)
    data['Hour'] = data['Date-Time'].dt.hour
    data['Minute'] = data['Date-Time'].dt.minute
    data = data.drop(
        data.loc[(data['Hour'] < 8) | (data['Hour'] > 16) | ((data['Hour'] == 16) & (data['Minute'] > 0))].index)
    data = data.drop(['Hour', 'Minute', 'Day'], axis=1)
    data = data.iloc[1:]
    train_data_list.append(data)

train_data = pd.concat(train_data_list, ignore_index=True)
date = pd.to_datetime(train_data['Date-Time'].dt.strftime('%Y/%m/%d'))
unique_date = pd.unique(date)
num_of_training_days = len(unique_date)
print('Training Set Num of Days: ', num_of_training_days)
print('Train Data Unique Date: ', unique_date)

df_train = open('train_data.txt', 'wb')
pickle.dump(train_data, df_train)
df_train.close()

for raw_data in test_raw_data_list:
    data = raw_data.drop(['#RIC', 'Domain', 'GMT Offset', 'Type', 'L1-BuyNo', 'L1-SellNo', 'L2-BuyNo', 'L2-SellNo',
                          'L3-BuyNo', 'L3-SellNo', 'L4-BuyNo', 'L4-SellNo', 'L5-BuyNo', 'L5-SellNo',
                          'L6-BuyNo', 'L6-SellNo', 'L7-BuyNo', 'L7-SellNo', 'L8-BuyNo', 'L8-SellNo',
                          'L9-BuyNo', 'L9-SellNo', 'L10-BuyNo', 'L10-SellNo'], axis=1)
    data['Date-Time'] = pd.to_datetime(data['Date-Time'],
                                       format='%Y-%m-%dT%H:%M:%S.%fZ').dt.round('{}s'.format(600))
    data = data.groupby(['Date-Time']).first().reset_index()
    data['Day'] = data['Date-Time'].dt.dayofweek
    data = data.drop(data.loc[(data['Day'] == 5) | (data['Day'] == 6)].index)
    data['Hour'] = data['Date-Time'].dt.hour
    data['Minute'] = data['Date-Time'].dt.minute
    data = data.drop(
        data.loc[(data['Hour'] < 8) | (data['Hour'] > 16) | ((data['Hour'] == 16) & (data['Minute'] > 0))].index)
    data = data.drop(['Hour', 'Minute', 'Day'], axis=1)
    data = data.iloc[1:]
    test_data_list.append(data)

test_data = pd.concat(test_data_list, ignore_index=True)
date = pd.to_datetime(test_data['Date-Time'].dt.strftime('%Y/%m/%d'))
unique_date = pd.unique(date)
num_of_test_days = len(unique_date)

done = True

print('Test Set Num of Days: ', num_of_test_days)
print('Test Data Unique Date: ', unique_date)

df_test = open('test_data.txt', 'wb')
pickle.dump(test_data, df_test)
df_test.close()
