"""Testing of the agent

Author: YANG, Austin Liu
Created Date: Feb. 26, 2017
Modified Date: Mar. 6 2017
"""


from zipline.api import (
    order,
    record,
    symbol,
    history,
    get_datetime,
    set_long_only)
import matplotlib.pyplot as plt
import pandas as pd
from global_values import (
    mylogger, directory_log,
    TP_matrixs,
    capital_base, mu,
    action_set, Q_function)


def initialize_test(context):
    # AAPL
    context.security = symbol('AAPL')

    # Algorithm will only take long positions.
    # It will stop if encounters a short position.
    set_long_only()


def handle_data_test(context, data):
    # Get current date
    now = str(get_datetime('US/Eastern'))[0:11] + "00:00:00+0000"

    # Get current state
    state = TP_matrixs.ix(now)

    # Choose the action of the highest Q-Value
    action_values = [Q_function(state, action_set[0]),
                     Q_function(state, action_set[1]),
                     Q_function(state, action_set[2])]
    action = action_set[action_values.index(max(action_values))]

    # Execute chosen action
    now = now[0: 10]
    if action == action_set[0]:
        # Sell
        # No short
        try:
            order(context.security, -mu)
            mylogger.logger.info(now + ': sell')
        except Exception as e:
            mylogger.logger.info(now + ': No short!')
    elif action == action_set[1]:
        # Buy
        # No cover
        if context.portfolio.cash >= data.current(context.security, 'price') * mu:
            order(context.security, mu)
            mylogger.logger.info(now + ': buy')
            mylogger.logger.info(context.portfolio.cash)
        else:
            mylogger.logger.info(now + ': No cover!')
    elif action == action_set[2]:
        # Hold
        mylogger.logger.info(now + ': hold')
        pass

    # Save values for later inspection
    record(AAPL=data.current(context.security, 'price'),
           actions=action)


def analyze_test(context=None, results=None):
    """Anylyze the result of algorithm"""
    # Total profit and loss
    total_pl = (results['portfolio_value'][-1] - capital_base) / capital_base
    mylogger.logger.info('Total profit and loss: ' + str(total_pl))

    # Hit rate by day
    hit_num = 0
    actions = results['actions'].dropna()
    actions = actions.drop(actions.index[-1])
    hit_record = actions.copy(deep=True)
    for date in hit_record.index:
        loc_current = results['AAPL'].index.get_loc(date)
        change = results['AAPL'][loc_current + 1] - \
            results['AAPL'][loc_current]
        # "hit" means that trend and signal match
        # "miss" means that trend and signal dismatch
        if (change > 0 and results['actions'][date] == 'buy')\
                or (change < 0 and results['actions'][date] == 'sell')\
                or (change == 0 and results['actions'][date] == 'hold'):
            hit_record[date] = 'hit'
            hit_num += 1
        else:
            hit_record[date] = 'miss'
    # compute hit rate
    hit_rate = hit_num / len(hit_record)
    # Construct hit table
    hit_data = {'signal': actions.values,
                'hit/miss': hit_record.values}
    hit_table = pd.DataFrame(hit_data, index=hit_record.index)
    mylogger.logger.info('Hit table:')
    mylogger.logger.info('Date          signal  hit/miss')
    for i in range(0, len(hit_table)):
        mylogger.logger.info(str(hit_table.index[i])[0: 10] + '    ' +
                             str(hit_table['signal'][i]) + '    ' +
                             str(hit_table['hit/miss'][i]))
    mylogger.logger.info('Hit number:' + str(hit_num) +
                         '/' + str(len(hit_record)))
    mylogger.logger.info('Hit rate:' + str(hit_rate))

    # Draw the figure
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.set_window_title('Q-Learning Stock Trading Algorithm')

    # Subplot 1
    # Comparison between portfolio value and stock value
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Comparison between Portfolio Value and Stock Value')
    # Portfolio value
    results['portfolio_value'].plot(ax=ax1,
                                    label='Portfolio')
    # Stock value with the same initialization
    stock_value = results['AAPL'].copy(deep=True)
    flag_first = True
    share_number = 0
    for day in stock_value.index:
        if flag_first:
            share_number = capital_base / stock_value[day]
            stock_value[day] = capital_base
            flag_first = False
        else:
            stock_value[day] *= share_number
    stock_value.plot(ax=ax1,
                     color='k',
                     label='APPL')
    plt.legend(loc='upper left')

    # Subplot 2
    # Marks of actions
    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('Action Marks')
    results['AAPL'].plot(ax=ax2,
                         color='k',
                         label='AAPL Price')
    actions_sell = results['actions'].ix[[
        action == 'sell' for action in results['actions']]]
    actions_buy = results['actions'].ix[[
        action == 'buy' for action in results['actions']]]
    actions_hold = results['actions'].ix[[
        action == 'hold' for action in results['actions']]]
    # Use "v" to represent sell action
    ax2.plot(actions_sell.index,
             results['AAPL'].ix[actions_sell.index],
             'v',
             markersize=2,
             color='g',
             label='Sell')
    # Use "^" to represent buy action
    ax2.plot(actions_buy.index,
             results['AAPL'].ix[actions_buy.index],
             '^',
             markersize=2,
             color='r',
             label='Buy')
    # Use "." to represent hold action
    ax2.plot(actions_hold.index,
             results['AAPL'].ix[actions_hold.index],
             '.',
             markersize=2,
             color='b',
             label='Hold')
    plt.legend(loc='upper left')

    # Save figure into file
    fig_name = 'log/' + directory_log + '/fig' + directory_log + '.png'
    plt.savefig(fig_name)

    # Show figure on the screen
    plt.show()
