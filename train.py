"""Training of the agent

Author: YANG, Austin Liu
"""


from zipline.api import (
    order,
    symbol,
    get_datetime)
import random
import numpy as np
import global_values as gv
import pdb


# Discount factor
gama = 0.95

# Data and labels of models
Q_data = {
    "sell": np.array([]),
    "buy": np.array([]),
    "hold": np.array([])}
Q_labels = {
    "sell": np.array([]),
    "buy": np.array([]),
    "hold": np.array([])}


def initialize(context):
    # AAPL
    context.security = symbol('AAPL')

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

    # Initialize previous date
    context.date_prev = ''

    # Initialize previous taken action
    context.action_prev = ''

    # Initialize previous portfolio
    context.portfolio_prev = gv.capital_base


def handle_data(context, data):
    # Get current date
    now = str(get_datetime('US/Eastern'))[0:11] + "00:00:00+0000"

    # Get current state
    state = gv.TP_matrixs.ix[now].values

    # Epsilon-greedy Algorithm
    # Choose an action to execute according to current state
    probab = random.random()
    if probab <= gv.epsilon:
        # Epsilon
        # Take random action
        action = gv.action_set[random.randint(0, 2)]
    else:
        # 1 - epsilon
        # Take the action of the highest Q-Value
        action_values = [gv.Q_function(state, gv.action_set[0]),
                         gv.Q_function(state, gv.action_set[1]),
                         gv.Q_function(state, gv.action_set[2])]
        action = gv.action_set[action_values.index(max(action_values))]

    # Execute chosen action
    if action == gv.action_set[0]:
        # Sell
        order(context.security, -gv.mu)
    elif action == gv.action_set[1]:
        # Buy
        order(context.security, gv.mu)
    elif action == gv.action_set[2]:
        # Hold
        pass

    # Construct labeled data
    global Q_data, Q_labels
    # Juage if it's the first day
    if context.action_prev != "":
        y = context.portfolio.portfolio_value - \
            context.portfolio_prev + gama * gv.Q_function(state, action)
        # Add new data
        if Q_data[context.action_prev].size == 0:
            Q_data[context.action_prev] = np.array(
                [gv.TP_matrixs.ix[context.date_prev].values.tolist()],
                dtype=np.float32)
        else:
            Q_data[context.action_prev] = np.vstack((
                Q_data[context.action_prev],
                [gv.TP_matrixs.ix[context.date_prev].values.tolist()]))
        # Add new label
        Q_labels[context.action_prev] = np.append(
            Q_labels[context.action_prev], y)

    # Update saved previous information
    context.date_prev = now
    context.action_prev = action
    context.portfolio_prev = context.portfolio.portfolio_value
