import numpy as np
from typing import Dict


class Estimator:
    def __init__(self, p0: Dict[float, float], y: np.array):
        """
        Inicjalizacja parametrów:
            p0 : rozkład a priori
            y  : pomiary
        """
        self.p = p0
        self.y = y

    def estimate(self):
        # Rozmiar wektora pomiarów
        y_size = np.size(self.y)

        # Macierz wykładników A
        apow = np.ones((y_size, y_size))
        apow[np.triu_indices(y_size, 0)] = 0
        apow = np.cumsum(apow, axis=0)

        # Macierz A dla a=0.8
        A = {0.8: 0.8 * np.ones((y_size, y_size))}
        A[0.8] = np.power(A[0.8], apow)
        A[0.8][np.triu_indices(y_size, 1)] = 0

        # Macierz A dla a=0.9
        A[0.9] = 0.9 * np.ones((y_size, y_size))
        A[0.9] = np.power(A[0.9], apow)
        A[0.9][np.triu_indices(y_size, 1)] = 0

        # P(a=0.8|X)
        p = {0.8: self.p[0.8] * normDist(self.y, 0, A[0.8]@A[0.8].T)}
        p[0.8] = p[0.8] / (self.p[0.9] * normDist(self.y, 0, A[0.9]@A[0.9].T) + self.p[0.8] * normDist(self.y, 0, A[0.8]@A[0.8].T))
        p[0.8] = np.float(p[0.8])

        # P(a=0.9|X)
        p[0.9] = self.p[0.9] * normDist(self.y, 0, A[0.9]@A[0.9].T)
        p[0.9] = p[0.9] / (self.p[0.9] * normDist(self.y, 0, A[0.9]@A[0.9].T) + self.p[0.8] * normDist(self.y, 0, A[0.8]@A[0.8].T))
        p[0.9] = np.float(p[0.9])

        return p


def normDist(y: np.ndarray, mi: np.ndarray, cov: np.ndarray):
     return (1 / np.sqrt(np.linalg.det(cov)) *
             np.exp(- 0.5 * np.transpose(y-mi) @ np.linalg.inv(cov) @ (y-mi)))
