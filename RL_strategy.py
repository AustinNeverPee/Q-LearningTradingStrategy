"""Q-Learning Stock Trading Algorithm
State: {p0, p1, ..., pt}
Action: {buy, sell, hold}
Reward: difference between current and previous portfolio value
Approximate function: Neural Network
Train agent for only one stock
Using epsilon-greedy algorithm to train

Author: YANG, Austin Liu
Created Date: Feb. 26, 2017
"""


import pytz
from datetime import datetime
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo
import os
from global_values import (
    mylogger, directory_log,
    capital_base,
    model_dirs, Q_data, Q_labels, cnn_model_fn,
    epsilon, action_set, date_prev, action_prev,
    portfolio_prev)
from train import (
    initialize_train,
    handle_data_train)
from test import (
    initialize_test,
    handle_data_test,
    analyze_test)
import numpy as np
from tensorflow.contrib import learn

import pdb


# Training steps
training_steps = 100
# Number of agent training
Q_training_iters = 100


def initialize_log():
    """Initialize log module"""
    # Create log directory
    os.makedirs('log/' + directory_log)

    # Add file handle to mylogger
    mylogger.addFileHandler(directory_log)


def load_data():
    """Load stock data
       Both training and testing data
    """
    # Load data manually from Yahoo! finance
    # Training data
    start_train_data = datetime(2010, 1, 1, 0, 0, 0, 0, pytz.utc)
    end_train_data = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    data_train = load_bars_from_yahoo(stocks=['AAPL'],
                                      start=start_train_data,
                                      end=end_train_data)

    # Testing data
    start_test_data = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
    end_test_data = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
    data_test = load_bars_from_yahoo(stocks=['AAPL'],
                                     start=start_test_data,
                                     end=end_test_data)

    return [data_train, data_test]


def initialize_params_train(iter):
    """Initialize parameters used in training"""
    # Initialize labeled data
    global Q_data, Q_labels
    Q_data = {
        "sell": np.array([]),
        "buy": np.array([]),
        "hold": np.array([])}
    Q_labels = {
        "sell": np.array([]),
        "buy": np.array([]),
        "hold": np.array([])}

    # Initialize saved previous information
    global date_prev, state_prev, portfolio_prev
    date_prev = ''
    state_prev = []
    portfolio_prev = capital_base

    # Update epsilon
    global epsilon
    epsilon = pow(epsilon, iter)


def Q_update():
    """Update weights of three models:
       "sell" model, "buy" model and "hold" model
    """
    for action in action_set:
        # Create the estimator
        Q_estimator = learn.Estimator(
            model_fn=cnn_model_fn,
            model_dir=model_dirs[action])

        # Train the model
        Q_estimator.fit(
            x=Q_data,
            y=Q_labels,
            steps=training_steps)


def agent_train(data_train):
    """Train the agent
       Learn from the environment
    """
    for iter in range(0, Q_training_iters):
        mylogger.logger.info('Agent Iteration :', iter + 1)

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


def agent_test(data_test):
    """Test the agent
       Check out the result of learning
    """
    # Create algorithm object passing in initialize and
    # handle_data functions
    algo_obj = TradingAlgorithm(initialize=initialize_test,
                                handle_data=handle_data_test,
                                analyze=analyze_test,
                                data_frequency='daily',
                                capital_base=capital_base)

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
