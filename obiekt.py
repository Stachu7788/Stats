import numpy as np
import matplotlib.pyplot as plt
from typing import List


class object:
    def __init__(self, param: List[float] = None,
                 tf: float = 3., ts: float = 0.01):
        if param is None:
            self.n = int(tf/ts+1)
            self.rand_param()
        else:
            self.n = len(param)
            self.param = param
        self.state = np.zeros([self.n]).tolist()
        self.output = np.zeros([self.n]).tolist()
        self.ts = ts
        self.tf = ts * self.n
        self.t = 0.

    def step(self):
        k = int(self.t/self.ts)
        w_k = np.random.normal()
        v_k = np.random.normal()
        if k != self.n:
            self.state[k+1] = self.param[k] * self.state[k] + w_k
        self.output[k] = self.state[k] + v_k
        self.t += self.ts
        return k

    def plot(self):
        T = np.linspace(0., self.t, self.n)
        plt.plot(T, self.state, label="Stan", lw=0.6)
        plt.plot(T, self.output, label="Wyjcie", lw=0.6)
        plt.legend()
        plt.grid()

    def __str__(self, k: int = None):
        if k is None:
            k = int((self.t-self.ts)/self.ts)
            time = self.t-self.ts
        else:
            time = self.ts * k
        x = self.state[k]
        y = self.output[k]
        return f'{round(time,3):4.3} -> Stan: {x:8.3}\t Wyjscie: {y:8.3}'

    def rand_param(self):
        self.param = np.random.randint(0, 10, self.n)
        self.param = np.where(self.param < 9, 0.8, 0.9)
