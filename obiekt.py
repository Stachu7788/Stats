import numpy as np
import matplotlib.pyplot as plt
from typing import List


class object:
    def __init__(self, param: List[float], ts: float = 0.01):
        self.state = np.zeros([len(param)]).tolist()
        self.output = np.zeros([len(param)]).tolist()
        self.ts = ts
        self.t = 0.
        self.param = param

    def step(self):
        k = int(self.t/self.ts)
        if len(self.param) < k:
            return None
        w_k = np.random.normal()
        v_k = np.random.normal()
        self.state[k+1] = self.param[k] * self.state[k] + w_k
        self.output[k] = self.state[k] + v_k
        self.t += self.ts

    def plot(self):
        T = np.linspace(0., self.t, int(self.t/self.ts)+1)
        plt.plot(T, self.state, label="Stan")
        plt.plot(T, self.output, label="Wyjcie")
        plt.legend()
        plt.grid()
