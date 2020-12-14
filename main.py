from obiekt import object
from estymator import matrix_I, matrix_A, pa, estimator
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

a = 100 * [.0]
ob = object(None, 0.1, 5)
ob.simulate()

x = np.linspace(-5, 5, 200)
pay = stat.multivariate_normal.pdf(x, 0, 2)
plt.plot(x, pay)