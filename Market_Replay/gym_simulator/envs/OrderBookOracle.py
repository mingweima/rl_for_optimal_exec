import pandas as pd
from datetime import datetime


class OrderBookOracle:
    COLUMNS = ['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG']

    # Oracle for reading historical exchange orders stream
    def __init__(self, start_time, end_time, orders_file_path):
        self.start_time = start_time
        self.end_time = end_time
        self.orders_file_path = orders_file_path
        self.orders_list = self.processOrders()

    def processOrders(self):
        def convertDate(date_str):
            try:
                return datetime.strptime(date_str, '%Y%m%d%H%M%S.%f')
            except ValueError:
                return convertDate(date_str[:-1])

        orders_df = pd.read_csv(self.orders_file_path)
        orders_df['TIMESTAMP'] = orders_df['TIMESTAMP'].astype(str).apply(convertDate)
        orders_df['SIZE'] = orders_df['SIZE'].astype(int)
        orders_df['PRICE'] = orders_df['PRICE'].astype(float)
        orders_df = orders_df.loc[(orders_df.TIMESTAMP >= self.start_time) & (orders_df.TIMESTAMP < self.end_time)]
        orders_list = orders_df.to_dict('records')

        return orders_list

