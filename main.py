from obiekt import object

a = 200 * [0.8] + 50 * [0.9] + 50 * [0.8]
ob = object()
for i in range(ob.n):
    ob.step()
    print(ob)
ob.plot()

