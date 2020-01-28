import pickle
from tqdm import tqdm
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--ticker', type=str, default='HSBA')
args = parser.parse_args()

ticker = args.ticker

months = ['2016-01-01_2016-01-31',
                    '2016-02-01_2016-02-29',
                    '2016-03-01_2016-03-31',
                    '2016-04-01_2016-04-30',
                    '2016-05-01_2016-05-31',
                    '2016-06-01_2016-06-30',
                    '2016-07-01_2016-07-31',
                    '2016-08-01_2016-08-31',
                    '2016-09-01_2016-09-30',
                    '2016-10-01_2016-10-31',
                    '2016-11-01_2016-11-30',
                    '2016-12-01_2016-12-31',
                    '2017-01-01_2017-01-31',
                    '2017-02-01_2017-02-28',
                    '2017-03-01_2017-03-31',
                    '2017-04-01_2017-04-30',
                    '2017-05-01_2017-05-31',
                    '2017-06-01_2017-06-30',
                    '2017-07-01_2017-07-31',
                    '2017-08-01_2017-08-31',
                    '2017-09-01_2017-09-30',
                    '2017-10-01_2017-10-31',
                    '2017-11-01_2017-11-30',
                    '2017-12-01_2017-12-31',
                    '2018-01-01_2018-01-31',
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
    path_name = '/nfs/home/mingweim/lob/{}/L2_{}.L_{}.csv.gz'.format(ticker, ticker, month)

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
                                       format='%Y-%m-%dT%H:%M:%S.%fZ').dt.round('{}s'.format(300))

    bar.update(1)
    bar.set_description('Grouping By -- {}'.format(month))

    data = data.groupby(['Date-Time']).first().reset_index()

    bar.update(1)
    bar.set_description('Deleting Weekends -- {}'.format(month))

    data['Day'] = data['Date-Time'].dt.dayofweek
    data = data.drop(data.loc[(data['Day'] == 5) | (data['Day'] == 6)].index)
    data = data.drop(data.loc[(data['Date-Time'] >= pd.to_datetime('2016/7/14'))
                              & (data['Date-Time'] < pd.to_datetime('2016/7/15'))].index)
    data = data.drop(data.loc[(data['Date-Time'] >= pd.to_datetime('2018/4/5'))
                              & (data['Date-Time'] < pd.to_datetime('2018/4/7'))].index)
    for year in [2016, 2017, 2018]:
        data = data.drop(data.loc[(data['Date-Time'] >= pd.to_datetime('{}/12/23'.format(year)))
                           & (data['Date-Time'] < pd.to_datetime('{}/12/29'.format(year)))].index)

    bar.update(1)
    bar.set_description('Deleting Auction Periods -- {}'.format(month))

    data['Hour'] = data['Date-Time'].dt.hour
    data['Minute'] = data['Date-Time'].dt.minute
    data = data.drop(
        data.loc[(data['Hour'] < 8) | (data['Hour'] > 16) | ((data['Hour'] == 16) & (data['Minute'] > 0))].index)
    data = data.drop(['Minute', 'Day'], axis=1)

    bar.update(1)
    bar.set_description('Storing Data -- {}'.format(month))

    df_train = open('/nfs/home/mingweim/rl_for_optimal_exec/'
                    'trading_environment/data/{}/{}.txt'.format(ticker, month), 'wb')
    pickle.dump(data, df_train)
    df_train.close()
    date = pd.to_datetime(data['Date-Time'].dt.strftime('%Y/%m/%d'))
    unique_date = pd.unique(date)
    for day in unique_date:
        for session in ['morning', 'afternoon']:
            df_train = open('/nfs/home/mingweim/rl_for_optimal_exec/trading_environment'
                            '/data/{}/{}_{}_{}.txt'.format(ticker, month, day, session), 'wb')
            if session == 'morning':
                session_data = data.loc[data['Date-Time'] >= day + pd.Timedelta('{}hours'.format(8))]
                session_data.reset_index(drop=True, inplace=True)
                session_data = session_data.iloc[:49]
                ext = session_data.loc[48]
                if ext['Hour'] != 12:
                    print('Day: ', day)
                    print(session_data)
            else:
                session_data = data.loc[data['Date-Time'] >= day + pd.Timedelta('{}hours'.format(11))]
                session_data.reset_index(drop=True, inplace=True)
                session_data = session_data.iloc[:49]
                ext = session_data.loc[48]
                if ext['Hour'] != 15:
                    print('Day: ', day)
                    print(session_data)

            pickle.dump(session_data, df_train)
            df_train.close()

    bar.update(1)
    bar.set_description('Finished Processing Data -- {}'.format(month))
    bar.close()
