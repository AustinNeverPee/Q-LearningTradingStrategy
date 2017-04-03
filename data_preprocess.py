"""Data Preprocessing
Preprocess raw price data into the form of TP Matrix

Author: YANG, Austin Liu
Created Date: Mar. 21, 2017
Modified Date: Apr. 3 2017
"""

import numpy as np
import pandas as pd

import pytz
from datetime import datetime
from zipline.utils.factory import load_bars_from_yahoo
import pdb


# Settings of TP Matrix
# Both are based on Fibonacci numbers
# Range of price changes ratio
rows = ((0, 2), (2, 5), (5, 10), (10, 18), (18, 31), (31, 52), (52, 86), (86, 141), (141, 100000), \
        (-2, 0), (-5, -2), (-10, -5), (-18, -10), (-31, -18), (-52, -31), (-86, -52), (-141, -86), (-100000, -141))
# Time window
columns = ((1, 2), (3, 5), (6, 10), (11, 18), (19, 31), (32, 52), (53, 86), (87, 141), (142, 230)) * 2


# Load stock 
def load_data():
    # Load data manually from Yahoo! finance
    start = datetime(2009, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    data = load_bars_from_yahoo(stocks = ['AAPL'],
                                start = start,
                                end = end)

    # Initialize TP Matrix
    # 3-dimension: # of stock * 18 * 18
    # narray
    _TP_matrixs = np.zeros((len(data.ix['AAPL']) - 230, 18, 18), dtype = np.bool)
    TP_matrixs = pd.Panel(_TP_matrixs, items = data.ix['AAPL'].index[230:])

    # Construct TP Matrix
    for TP_matrix in TP_matrixs.iteritems():
        # Extract raw close price of last 230 days
        _list_CP = data.ix['AAPL'][data.ix['AAPL'].index < TP_matrix[0]]['close'].tolist()
        list_CP = _list_CP[len(_list_CP) - 230 : len(_list_CP)]

        # col[0, 8] for Upward TP Matrix
        # col[9, 17] for Downward TP Matrix
        for col in range(0, 18):
            D = columns[col][0] - 1
            for row in range(0, 18):
                # For each element of TP Matrix
                for TP in range(D, columns[col][1]):
                    # Change ratio of stock on day D with repect to the price at TP
                    C_TPD = (list_CP[TP] - list_CP[D]) / list_CP[D]
                    if C_TPD * 100 >= rows[row][0] and C_TPD * 100 < rows[row][1]:
                        TP_matrix[1][row][col] = True
                        break
        print(TP_matrix[0])
        print(TP_matrix[1])
    pdb.set_trace()

    return data


if __name__ == '__main__':
    # Load stock data
    data = load_data()
    print(type(data))
    print(data)
