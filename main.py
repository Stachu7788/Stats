from obiekt import object
from estymator import estimator

a = 100 * [.0]
ob = object(a, 0.1, 20)
ob.simulate()
ob.plot()


