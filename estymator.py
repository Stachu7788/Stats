import numpy as np
from obiekt import object


class estimator(object):
    def __init__(self, *args):
        object.__init__(self, *args)
        self.est = np.zeros([self.n]).tolist()
