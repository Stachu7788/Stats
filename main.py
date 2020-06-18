from obiekt import object
from estymator import *

a = 100 * [.0]
ob = object(None, 0.1, 5)
ob.simulate()
ob.plot()

x=ob.output
k=len(x)
a=0.8
A = matrix_A(a, k)
A8 = matrix_A(0.8, k)
A9 = matrix_A(0.9, k)
I_ = matrix_I(k)
pay = stat.multivariate_normal.pdf(x, 0, A @ A.T + I_)