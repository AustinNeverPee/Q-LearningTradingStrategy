"""Log module

Author: YANG, Austin Liu
"""


import logging


class MyLogger:
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    formatter_ch = logging.Formatter('%(message)s')
    formatter_fh = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def __init__(self):
        # Screen log
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(self.formatter_ch)
        self.logger.addHandler(ch)

    def addFileHandler(self, directory_log):
        # File log
        filename = 'log/' + directory_log + '/log' + directory_log + '.log'
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.formatter_fh)
        self.logger.addHandler(fh)
