import numpy as np
import scipy.stats as stat
from obiekt import object


class estimator(object):
    def __init__(self, *args):
        object.__init__(self, *args)
        self.est = np.zeros([self.n]).tolist()

    def step(self):
        k = object.step()
        x = np.array(self.state[:k+1])      # n, 1
        y = np.array(self.output[:k+1])     # n, 1
        a = np.array(self.est[:k])          # n-1, 1
        


def matrix_A(a, k):
    A = np.ones([k, k])
    A = np.tril(A)
    b = np.array([1. for i in range(k)]).reshape([k, 1])
    for i in range(1, k):
        A = np.concatenate((b, A), axis=1)
        np.fill_diagonal(A, a**i)
    A = A[:, k-1:]
    return A


def matrix_I(k):
    I_ = np.diag(k * [1.])
    return I_


def pa(a):
    return 0.9 if a == 0.8 else 0.1


def normal_dist(y, a, k):
    A = matrix_A(a, k)
    A8 = matrix_A(0.8, k)
    A9 = matrix_A(0.9, k)
    I_ = matrix_I(k)
    pay = ((stat.multivariate_normal.pdf(y, 0, A @ A.T + I_) * pa(a)) /
           (pa(0.8) * stat.multivariate_normal.pdf(y, 0, A8 @ A8.T + I_) +
            pa(0.9) * stat.multivariate_normal.pdf(y, 0, A9 @ A9.T + I_)))
    return pay
