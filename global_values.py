"""Global values used in the algorithm
Variables and functions

Author: YANG, Austin Liu
Created Date: Feb. 26, 2017
"""


from datetime import datetime
from pybrain.structure import *
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pickle
from log import MyLogger


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
# Previous portfolio
portfolio_prev = capital_base
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                            END                              *'
'*                          Trading                            *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                         Q-Learning                          *'
'*                           START                             *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
# Parameter used in Epsilon-greedy Algorithm
epsilon = 0.955
# Discount factor
gama = 0.95
# Action set
# Contain all the possible actions to execute in each state
action_set = ['sell', 'buy', 'hold']
# Previous state
state_prev = []
# Previous taken action
action_prev = ''
# Q-function
# Use nueral network to estimate


def Q_function(state, action):
    if action == "sell":
        return net_sell.activate(state)
    elif action == "buy":
        return net_buy.activate(state)
    elif action == "hold":
        return net_hold.activate(state)


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                            END                              *'
'*                         Q-Learning                          *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'


'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                           Model                             *'
'*                           START                             *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
# Number of training
training_iters = 1
# Maximum iterations for network to converge
epochs = 100
# Update rate of neural network
alpha = 0.01
# Three nueral networks to simulate sell, buy and hold Q-function
# Network of "sell"
net_sell = FeedForwardNetwork()
in_layer_sell = LinearLayer(20, name='in_layer_sell')
hidden_layer_sell = SigmoidLayer(30, name='hidden_layer_sell')
out_layer_sell = LinearLayer(1, name='out_layer_sell')
net_sell.addInputModule(in_layer_sell)
net_sell.addModule(hidden_layer_sell)
net_sell.addOutputModule(out_layer_sell)
in_to_hidden_sell = FullConnection(in_layer_sell, hidden_layer_sell)
hidden_to_out_sell = FullConnection(hidden_layer_sell, out_layer_sell)
net_sell.addConnection(in_to_hidden_sell)
net_sell.addConnection(hidden_to_out_sell)
net_sell.sortModules()
# Network of "buy"
net_buy = FeedForwardNetwork()
in_layer_buy = LinearLayer(20, name='in_layer_buy')
hidden_layer_buy = SigmoidLayer(30, name='hidden_layer_buy')
out_layer_buy = LinearLayer(1, name='out_layer_buy')
net_buy.addInputModule(in_layer_buy)
net_buy.addModule(hidden_layer_buy)
net_buy.addOutputModule(out_layer_buy)
in_to_hidden_buy = FullConnection(in_layer_buy, hidden_layer_buy)
hidden_to_out_buy = FullConnection(hidden_layer_buy, out_layer_buy)
net_buy.addConnection(in_to_hidden_buy)
net_buy.addConnection(hidden_to_out_buy)
net_buy.sortModules()
# Network of "hold"
net_hold = FeedForwardNetwork()
in_layer_hold = LinearLayer(20, name='in_layer_hold')
hidden_layer_hold = SigmoidLayer(30, name='hidden_layer_hold')
out_layer_hold = LinearLayer(1, name='out_layer_hold')
net_hold.addInputModule(in_layer_hold)
net_hold.addModule(hidden_layer_hold)
net_hold.addOutputModule(out_layer_hold)
in_to_hidden_hold = FullConnection(in_layer_hold, hidden_layer_hold)
hidden_to_out_hold = FullConnection(hidden_layer_hold, out_layer_hold)
net_hold.addConnection(in_to_hidden_hold)
net_hold.addConnection(hidden_to_out_hold)
net_hold.sortModules()
# Three training sets for three different networks
data_train_sell = SupervisedDataSet(20, 1)
data_train_buy = SupervisedDataSet(20, 1)
data_train_hold = SupervisedDataSet(20, 1)
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
'*                            END                              *'
'*                           Model                             *'
'*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*'
