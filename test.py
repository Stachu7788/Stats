from estymator import Estimator
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


def measurements(a_val: float or List, length: int):
    x = np.zeros([length, 1])
    for i in range(1, length):
        x[i, 0] = a_val * x[i-1, 0] + np.random.normal()
    return x

def mean(est: List[Dict[float, float]]):
    mean = {}
    mean[0.8] = np.mean([estimation[0.8] for estimation in est])
    mean[0.9] = np.mean([estimation[0.9] for estimation in est])
    return mean

def variance(est: List[Dict[float, float]]):
    var = {}
    var[0.8] = np.var([estimation[0.8] for estimation in est])
    var[0.9] = np.var([estimation[0.9] for estimation in est])
    return var

def wrong_estimations(est: List[Dict[float, float]], a: float):
    counter = 0
    for estimation in est:
        if estimation[a] < 0.5:
            counter += 1
    return counter

def sample_length():
    samples = {}
    lengths = [50, 100, 200, 350, 600]
    for n in lengths:
        samples[n] = []
        for _ in range(30):
            y = measurements(0.8, n)
            samples[n].append(Estimator({0.8: 0.9, 0.9: 0.1}, y).estimate())
    for n in lengths:
        print(f"Długość pomiaru: {n}")
        print(f"Wartość oczekiwana: {mean(samples[n])}")
        print(f"Wariancja: {variance(samples[n])}")
        print(f"Liczba błędnych estymacji(na 30): {wrong_estimations(samples[n], 0.8)}")
        for sample in samples[n]:
            plt.scatter(n, sample[0.8], c='b', marker='.')
    plt.xlabel("Długoś wektora pomiarów")
    plt.ylabel("Prawdopodobieństwo a=0.8")
    plt.title("Zależność prawdopodobieństwa od liczby pomiarów")
    plt.show()
    plt.savefig("a.png")

def a_priori_prob():
    cases = [{0.8: 0.9, 0.9: 0.1},{0.8: 0.75, 0.9: 0.25},
             {0.8: 0.5, 0.9: 0.5},{0.8: 0.25, 0.9: 0.75},{0.8: 0.1, 0.9: 0.9}]
    samples = {}
    for n in range(len(cases)):
        samples[n] = []
        for _ in range(30):
            y = measurements(0.8, 80)
            samples[n].append(Estimator(cases[n], y).estimate())
    for n in range(len(cases)):
        print(f"Prawdopodobieństwo a priori: {cases[n]}")
        print(f"Wartość oczekiwana: {mean(samples[n])}")
        print(f"Wariancja: {variance(samples[n])}")
        print(f"Liczba błędnych estymacji(na 30): {wrong_estimations(samples[n], 0.8)}")
        for sample in samples[n]:
            plt.scatter(n, sample[0.8], c='b', marker='.')
    plt.xlabel("Prawdopodobieństwo a priori")
    plt.ylabel("Prawdopodobieństwo a=0.8")
    plt.title("Zależność prawdopodobieństwa posteriori od a priori p(0.8)")
    plt.xticks([n for n in range(len(cases))], [p[0.8] for p in cases])
    plt.show()
    plt.savefig("b.png")
sample_length()
a_priori_prob()