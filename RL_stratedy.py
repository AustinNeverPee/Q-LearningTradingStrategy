"""Q-Learning Stock Trading Algorithm
State: {p0, p1, ..., pt}
Action: {buy, sell, hold}
Reward: difference between current and previous portfolio value
Approximate function: Neural Network
Train agent for only one stock
Using epsilon-greedy algorithm to train

Author: YANG, Austin Liu
Created Date: Feb. 26, 2017
Modified Date: Mar. 6 2017
"""


import pytz
from datetime import datetime
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo
import random
import os

from utils import *
from train import *
from test import *
from log import MyLogger


# Initialize log module
def initialize_log():
    # Create log directory
    os.makedirs('log/' + directory_log)

    # Add file handle to mylogger
    mylogger.addFileHandler(directory_log)


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


# Train the agent
# Learn from the environment 
def agent_train(data_train):
    for iter in range(0, training_iters):
        print('Iteration :', iter + 1)

        # Initialize parameters used in training
        initialize_params_train(iter)
        
        # Create algorithm object passing in initialize, 
        # handle_data functions and so on
        algo = TradingAlgorithm(initialize=initialize_train, 
                                handle_data=handle_data_train,
                                data_frequency='daily',
                                capital_base=capital_base)
        
        # Run algorithm
        perf = algo.run(data_train)

        # Train neural network with produced training set
        Q_update()


# Test the agent
# Check out the result of learning
def agent_test(data_test):
    # Create algorithm object passing in initialize and
    # handle_data functions
    algo_obj = TradingAlgorithm(initialize=initialize_test, 
                                handle_data=handle_data_test,
                                analyze = analyze_test,
                                data_frequency='daily',
                                capital_base = capital_base)
    
    # Run algorithm
    perf = algo_obj.run(data_test)


if __name__ == '__main__':
    # Initialize log module
    initialize_log()
    
    # Load stock data
    [data_train, data_test] = load_data()

    # Train the agent
    agent_train(data_train)

    # Test the agent
    agent_test(data_test)
