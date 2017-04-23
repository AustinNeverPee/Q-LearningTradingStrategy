"""Q-Learning Stock Trading Algorithm
State: TP Matrix
Action: {buy, sell, hold}
Reward: difference between current and previous portfolio value
Approximate function: Convolutional Neural Network
Train agent for only one stock
Using epsilon-greedy algorithm to train

Author: YANG, Austin Liu
"""


import pytz
from datetime import datetime
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo
import os
import global_values as gv
import train
import test
import numpy as np
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
import pdb


# tf.logging.set_verbosity(tf.logging.INFO)


# Training steps
training_steps = 100
# Number of agent training
Q_training_iters = 5


def initialize_log():
    """Initialize log module"""
    # Create log directory
    os.makedirs('log/' + gv.directory_log)

    # Add file handle to mylogger
    gv.mylogger.addFileHandler(gv.directory_log)


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


def Q_update():
    """Update weights of three models:
       "sell" model, "buy" model and "hold" model
    """
    for action in gv.action_set:
        gv.mylogger.logger.info("Update " + action + " model")

        # # Configure a ValidationMonitor with training data
        # validation_monitor = learn.monitors.ValidationMonitor(
        #     np.float32(Q_data[action]),
        #     np.float32(Q_labels[action]),
        #     every_n_steps=20)

        # Create the estimator
        Q_estimator = learn.Estimator(
            model_fn=gv.cnn_model_fn,
            model_dir=gv.model_dirs[action])

        # Train the model
        SKCompat(Q_estimator).fit(
            x=train.Q_data[action].astype(np.float32),
            y=train.Q_labels[action].astype(np.float32),
            steps=training_steps)

        # Evaluate the model and print results
        eval_results = Q_estimator.evaluate(
            x=train.Q_data[action].astype(np.float32),
            y=train.Q_labels[action].astype(np.float32))
        gv.mylogger.logger.info(eval_results)


def agent_train(data_train):
    """Train the agent
       Learn from the environment
    """
    for iter in range(0, Q_training_iters):
        gv.mylogger.logger.info("Agent Iteration :" + str(iter + 1))

        # Create algorithm object passing in initialize,
        # handle_data functions and so on
        algo = TradingAlgorithm(initialize=train.initialize,
                                handle_data=train.handle_data,
                                data_frequency='daily',
                                capital_base=gv.capital_base)

        # Run algorithm
        perf = algo.run(data_train)

        # Train neural network with produced training set
        Q_update()

        # Update epsilon
        gv.epsilon = pow(gv.epsilon, iter + 2)


def agent_test(data_test):
    """Test the agent
       Check out the result of learning
    """
    # Create algorithm object passing in initialize and
    # handle_data functions
    algo_obj = TradingAlgorithm(initialize=test.initialize,
                                handle_data=test.handle_data,
                                analyze=test.analyze,
                                data_frequency='daily',
                                capital_base=gv.capital_base)

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
