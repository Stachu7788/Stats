from obiekt import object

a = 200 * [0.8] + 50 * [0.9] + 50 * [0.8]
ob = object(a)
for i in a:
    ob.step()
ob.plot()

