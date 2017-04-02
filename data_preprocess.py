"""Data Preprocessing
Preprocess raw price data into the form of TP Matrix

Author: YANG, Austin Liu
Created Date: Mar. 21, 2017
"""


import pytz
from datetime import datetime
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo


# Load stock data
# Both training and testing data
def load_data():
    # Load data manually from Yahoo! finance
    # Training data
    start_train = datetime(2010, 1, 1, 0, 0, 0, 0, pytz.utc)
    end_train = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    data_train = load_bars_from_yahoo(stocks=['AAPL'],
                                      start=start_train,
                                      end=end_train)

    # Testing data
    start_test = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    end_test = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
    data_test = load_bars_from_yahoo(stocks=['AAPL'],
                                     start=start_test,
                                     end=end_test)

    return [data_train, data_test]


if __name__ == '__main__':
    # Load stock data
    [data_train, data_test] = load_data()
    print(type(data_test))
    print(data_test)
