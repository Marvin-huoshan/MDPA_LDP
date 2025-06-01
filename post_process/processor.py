import abc
import logging


class Processor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, n, users, fo):
        self.logger = logging.getLogger('Processor')
        self.n = n
        self.users = users
        self.fo = fo


    @abc.abstractmethod
    def calibrate(self, est_dist):
        return
