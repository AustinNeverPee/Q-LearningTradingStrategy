"""Global values used in the algorithm
Variables and functions

Author: YANG, Austin Liu
Created Date: Feb. 26, 2017
"""


from datetime import datetime
from os import (
    getcwd,
    path)
from random import uniform
import pickle
from log import MyLogger
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python import SKCompat

import pdb


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                            Log                              *'
'*                           START                             *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
# Instantiate log
mylogger = MyLogger()
# Log directory
directory_log = str(datetime.now())[0:19]
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                            END                              *'
'*                            Log                              *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                         TP Matrix                           *'
'*                           START                             *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
# Load TP Matrix from pickle file
pkl_file = open('TP_matrix.pkl', 'rb')
TP_matrixs = pickle.load(pkl_file)
pkl_file.close()
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                            END                              *'
'*                         TP Matrix                           *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                          Trading                            *'
'*                           START                             *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
# Starting cash
capital_base = 100000
# Largest trading amount allowed
mu = 100
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                            END                              *'
'*                          Trading                            *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                           Model                             *'
'*                           START                             *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
# Model directory
model_dirs = {
    "sell": getcwd() + "/model/sell_convnet_model",
    "buy": getcwd() + "/model/buy_convnet_model",
    "hold": getcwd() + "/model/hold_convnet_model"}
# Data and labels of models
Q_data = {
    "sell": np.array([], dtype=np.float32),
    "buy": np.array([], dtype=np.float32),
    "hold": np.array([], dtype=np.float32)}
Q_labels = {
    "sell": np.array([], dtype=np.float32),
    "buy": np.array([], dtype=np.float32),
    "hold": np.array([], dtype=np.float32)}


def cnn_model_fn(features, labels, mode):
    """Model function for CNN.
       CNN model to simulate sell, buy and hold Q-function
       Three models with the same structure
    """
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Each TP_matrix can be regarded as 18x18 image with 1 color channel
    input_layer = tf.reshape(features, [-1, 18, 18, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 18, 18, 1]
    # Output Tensor Shape: [batch_size, 18, 18, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 18, 18, 32]
    # Output Tensor Shape: [batch_size, 9, 9, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 9, 9, 32]
    # Output Tensor Shape: [batch_size, 9, 9, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 9, 9, 64]
    # Output Tensor Shape: [batch_size, 5, 5, 64]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        padding='same')

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 5, 5, 64]
    # Output Tensor Shape: [batch_size, 5 * 5 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 5 * 5 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == learn.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 1]
    logits = tf.layers.dense(
        inputs=dropout,
        units=1,
        name="logits")

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    # Mean Square Error
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(
            labels=tf.reshape(labels, [-1, 1]),
            predictions=logits)

    # Configure the Training Op (for TRAIN mode)
    # Adam Optimizer
    # Update rate of CNN
    alpha = 0.001
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=alpha,
            optimizer="Adam")

    # Generate Predictions
    predictions = {
        "results": logits
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                            END                              *'
'*                           Model                             *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                         Q-Learning                          *'
'*                           START                             *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
# Parameter used in Epsilon-greedy Algorithm
epsilon = 0.955
# Action set
# Contain all the possible actions to execute in each state
action_set = ['sell', 'buy', 'hold']
# Previous date
date_prev = []
# Previous taken action
action_prev = ''
# Previous portfolio
portfolio_prev = capital_base


def Q_function(state, action):
    """Q-function
       Use trained models to predict
    """
    if path.exists(getcwd() + "/model"):
        # If model already exists, then use the model to predict
        # Create the estimator
        Q_estimator = SKCompat(learn.Estimator(
            model_fn=cnn_model_fn,
            model_dir=model_dirs[action]))

        # Predict using the estimator
        predictions = Q_estimator.predict(x=state)

        return predictions["results"][0]
    else:
        # If model doesn't exist, just return random value
        return uniform(-10000, 10000)


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                            END                              *'
'*                         Q-Learning                          *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
