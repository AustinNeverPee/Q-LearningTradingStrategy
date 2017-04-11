"""Training of the agent

Author: YANG, Austin Liu
Created Date: Feb. 26, 2017
Modified Date: Mar. 6 2017
"""


from zipline.api import (
    order,
    symbol,
    get_datetime)
import random
import numpy as np
from global_values import (
    directory_log,
    TP_matrixs,
    mu,
    Q_data, Q_labels,
    epsilon, action_set, date_prev, action_prev, Q_function,
    portfolio_prev)
import pdb


# Discount factor
gama = 0.95


def initialize_train(context):
    # AAPL
    context.security = symbol('AAPL')

    print(directory_log)


def handle_data_train(context, data):
    pdb.set_trace()

    # Get current date
    now = str(get_datetime('US/Eastern'))[0:11] + "00:00:00+0000"

    # Get current state
    state = TP_matrixs.ix(now)

    # Epsilon-greedy Algorithm
    # Choose an action to execute according to current state
    probab = random.random()
    if probab <= epsilon:
        # Epsilon
        # Take random action
        action = action_set[random.randint(0, 2)]
    else:
        # 1 - epsilon
        # Take the action of the highest Q-Value
        action_values = [Q_function(state, action_set[0]),
                         Q_function(state, action_set[1]),
                         Q_function(state, action_set[2])]
        action = action_set[action_values.index(max(action_values))]

    # Execute chosen action
    if action == action_set[0]:
        # Sell
        order(context.security, -mu)
    elif action == action_set[1]:
        # Buy
        order(context.security, mu)
    elif action == action_set[2]:
        # Hold
        pass

    # Construct labeled data
    global date_prev, action_prev, portfolio_prev
    y = context.portfolio.portfolio_value - \
        portfolio_prev + gama * Q_function(state, action)
    global Q_data, Q_labels
    # Add new data
    if Q_data[action_prev].size:
        Q_data[action_prev] = np.array([TP_matrixs.ix(date_prev)])
    else:
        Q_data[action_prev] = np.vstack(
            (Q_data[action_prev], [TP_matrixs.ix(date_prev)]))
    # Add new label
    Q_labels[action_prev] = np.append(Q_labels[action_prev], y)

    # Update saved previous information
    date_prev = now
    action_prev = action
    portfolio_prev = context.portfolio.portfolio_value
