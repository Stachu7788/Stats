import numpy as np
import matplotlib.pyplot as plt
from typing import List


class object:
    def __init__(self, param: List[float] = None,
                 ts: float = 0.01, tf: float = 3.):
        if param is None:
            self.n = int(round(tf/ts)+1)
            self.rand_param()
        else:
            self.n = len(param)
            self.param = param
        self.state = np.zeros([self.n]).tolist()
        self.output = np.zeros([self.n]).tolist()
        self.ts = ts
        self.tf = ts * self.n
        self.k = 1

    def step(self):
        k = self.k
        w_k = np.random.normal()
        v_k = np.random.normal()
        if k < self.n - 1:
            self.state[k+1] = self.param[k] * self.state[k] + w_k
        if k <= self.n - 1:
            self.output[k] = self.state[k] + v_k
        self.k += 1
        return k

    def plot(self):
        T = np.linspace(0., self.tf, self.n)
        plt.plot(T, self.state, label="Stan", lw=0.6)
        plt.plot(T, self.output, label="Wyjscie", lw=0.6)
        plt.legend()
        plt.grid()
        plt.show()

    def __str__(self):
        k = self.k - 1
        time = k * self.ts
        x = self.state[k]
        y = self.output[k]
        return f'{round(time,3):4.3} -> Stan: {x:8.3}\t Wyjscie: {y:8.3}'

    def rand_param(self):
        self.param = np.random.randint(0, 10, self.n)
        self.param = np.where(self.param < 9, 0.8, 0.9)

    def simulate(self):
        for i in range(self.n):
            self.step()
