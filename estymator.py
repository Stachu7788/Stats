import numpy as np
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
    
#c = np.array([0.8**n for n in range(0, k)]).reshape([k, 1])
A = matrix_A(0.8,6)