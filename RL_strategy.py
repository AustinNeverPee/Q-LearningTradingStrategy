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
from global_values import *
from train import *
from test import *
from log import MyLogger

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import pdb


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


# Initialize parameters used in training
def initialize_params_train(iter):
    # Clear training set
    global data_train_sell, data_train_buy, data_train_hold
    data_train_sell.clear()
    data_train_buy.clear()
    data_train_hold.clear()

    # Initialize saved previous information
    global action_prev, state_prev, portfolio_prev
    action_prev = ''
    state_prev = []
    portfolio_prev = capital_base

    # Update epsilon
    global epsilon
    epsilon = pow(epsilon, iter)


# Update weights of three networks:
# "sell" network, "buy" network and "hold" network
def Q_update():
    def main(unused_argv):
        # Load training and eval data
        mnist = learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        # Create the Estimator
        mnist_classifier = learn.Estimator(
            model_fn=cnn_model_fn, model_dir="/Users/austin/Documents/MSc/Second Semester/Independent Project/stage work/April/CNN/tmp/mnist_convnet_model")

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        mnist_classifier.fit(
            x=train_data,
            y=train_labels,
            batch_size=100,
            steps=100,
            monitors=[logging_hook])

        # Configure the accuracy metric for evaluation
        metrics = {
            "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
        }

        # Evaluate the model and print results
        eval_results = mnist_classifier.evaluate(
            x=eval_data, y=eval_labels, metrics=metrics)
        print(eval_results)

    try:
        print('%dst run!!!!' % (i))
        tf.app.run()
    except:
        pass

    global net_sell, net_buy, net_hold
    # In case that training set has not enough data
    # pdb.set_trace()
    try:
        # Train network of "sell"
        print('sell network')
        trainer_sell = BackpropTrainer(
            net_sell, data_train_sell, verbose=True, learningrate=alpha)
        trainer_sell.trainUntilConvergence(maxEpochs=epochs)

        # Train network of "buy"
        print('buy network')
        trainer_buy = BackpropTrainer(
            net_buy, data_train_buy, verbose=True, learningrate=alpha)
        trainer_buy.trainUntilConvergence(maxEpochs=epochs)

        # Train network of "hold"
        print('hold network')
        trainer_hold = BackpropTrainer(
            net_hold, data_train_hold, verbose=True, learningrate=alpha)
        trainer_hold.trainUntilConvergence(maxEpochs=epochs)
    except Exception as e:
        pass


# Train the agent
# Learn from the environment
def agent_train(data_train):
    for iter in range(0, training_iters):
        print('Iteration :', iter + 1)

        # Initialize parameters used in training
        initialize_params_train(iter)

        # Create algorithm object passing in initialize,
        # handle_data functions and so on
        # start_train = datetime(2011, 1, 1, 0, 0, 0, 0, pytz.utc)
        # end_train = datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc)
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
    # start_test = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)
    # end_test = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
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
