# -*- coding: utf-8 -*-
"""
Plik źródłowy z rozwiązaniem zadania projektowego

@author: Michał Dydo, Adam Kalota, Michał Gorczyca, Mariusz Kuchta
        gr. 2, Śr. 14.45-16.15

Temat: Projekt estymatora parametru błędu z wykorzystaniem twierdzenia Bayesa

Treść zadania:
    x(k+1)=a*x(k)+w(k), w(k)~N(0,1), a=0.8-normalna praca, a=0.9-awaria,
    korzystając z twierdzenia Bayesa, zaprojektować estymator parametru
    a (detektor awarii), oszacować prawdopodobieństwo błędu, wyznaczyć
    rozkład estymatora dla a, wykonać odpowiednie symulacje potwierdzające
    działanie estymatora. x(0)=0; p(X|a)*p(a)=p(X,a)=p(a|X)*p(X)
"""

from typing import Dict
t_distribution = Dict[float, float]

import numpy as np

class BayesianEstimator():
    
    def __init__(self, p0: Dict[float,float] = {0.8: 0.5, 0.9:0.5}, y : np.array = None):
        """
        Inicjalizacja zmiennych. Na początku musimy znać rozkład priori parametru a.
        Rozkład ten musi zostać zainicjalizowany (tutaj, lub w funkcji estimate)
        
        Parametry:
            p0 : rozkład a priori parametru a
            y  : pomiary
        """
        self.__p = p0
        self.__y = y
        
        
    @staticmethod
    def N(x: float, m: float, S: float) -> float:
        """
        """
        return 1 / np.sqrt( np.linalg.det(S) ) * np.exp(- 0.5 * np.transpose(x-m) @ np.linalg.inv(S) @ (x-m) ) 
        
    def __estimate(self) -> Dict[float,float]:
        """
        Metoda implementująca estymator Bayesa.
        """
        
        # Aby utworzyć odpowiednie macierze trzeba znać rozmiar wektora pomiarów
        y_size = np.size(self.__y)
    
        # Tworzona jest macierz zawierająca wykładniki potęg macierzy A
        A_pow = np.ones((y_size, y_size), dtype=float)
        A_pow[np.triu_indices(y_size, 0)] = 0
        A_pow = np.cumsum(A_pow,axis=0)
        
        # Tworzona jest macierz A dla a=0.9
        A = { 0.9 : 0.9 * np.ones((y_size, y_size), dtype=float) }
        A[0.9][np.triu_indices(y_size, 1)] = 0
        A[0.9] = np.power(A[0.9], A_pow)
        A[0.9][np.triu_indices(y_size, 1)] = 0
        
        # Tworzona jest macierz A dla a=0.8
        A[0.8] = 0.8 * np.ones((y_size, y_size), dtype=float)
        A[0.8][np.triu_indices(y_size, 1)] = 0
        A[0.8] = np.power(A[0.8], A_pow)
        A[0.8][np.triu_indices(y_size, 1)] = 0
        
        # Na podstawie pomiarów wyznaczane jest P(a=0.9|X)
        p = { 0.9 : self.__p[0.9] * BayesianEstimator.N(self.__y,0,A[0.9]@A[0.9].T) }
        p[0.9] = p[0.9] / (self.__p[0.9] * BayesianEstimator.N(self.__y,0,A[0.9]@A[0.9].T) + self.__p[0.8] * BayesianEstimator.N(self.__y,0,A[0.8]@A[0.8].T))
        p[0.9] = np.float(p[0.9])
        
        # Na podstawie pomiarów wyznaczane jest P(a=0.8|X)
        p[0.8] = self.__p[0.8] * BayesianEstimator.N(self.__y,0,A[0.8]@A[0.8].T)
        p[0.8] = p[0.8] / (self.__p[0.9] * BayesianEstimator.N(self.__y,0,A[0.9]@A[0.9].T) + self.__p[0.8] * BayesianEstimator.N(self.__y,0,A[0.8]@A[0.8].T))
        p[0.8] = np.float(p[0.8])
        
        return {0.8: p[0.8], 0.9:p[0.9]}
    
    def estimate(self, p: Dict[float,float] = None, y : np.array = None) -> Dict[float,float]:
        """
        Zwraca estymacje rozkładu parametru a dla danych pomiarów.
        Estymacja jest wyznaczana na podstawie wszystkich przekazanych pomiarów
        
        Parametry:
            p : rozkład a priori parametru a
            y : pomiary
            
        Zwraca:
            estymacja rozkładu parametru a
        """
        if p is not None and y is not None:        
            self.__p = p
            self.__y = y
            
        if self.__p is None or self.__y is None:
            raise ValueError("Niezainicjalizowany estymator!")
            
        return self.__estimate()
    

if __name__ == "__main__":
    
    print("Plik z klasą implementującą estymator Bayesa \n")
    print("Przykładowe działanie estymatora dla 100 pomiarów, rozkładu \'a priori\' {0.8:0.5, 0.9:0.5} i a=0.9: \n")
    
    a = 0.9
    x_size = 100
    x0 = 0
    p0 = {0.8:0.5, 0.9:0.5}
    
    x = [x0]    
    for i in range(x_size):
        x.append(a*x[-1] + np.random.normal(0,1))        
    x = np.array(x)
    x = np.resize(x, (x_size,1))
    
    est = BayesianEstimator(p0,x)
    
    print(est.estimate())
    
  