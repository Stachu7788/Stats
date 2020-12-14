# -*- coding: utf-8 -*-
"""
Testy zaprojektowanego estymatora. Estymator sprawdzany jest dla różnej 
wielkosci wektora danych oraz dla różnych, przyjętych rozkładów 'a priori'
parametru a. Dla każdego scenariusza testowego wykonywanych jest 10 prób.
Można zmienić ilosc tych iteracji poprzez zmianę wartosci parametru [iterations].

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

from typing import Dict, List

from source import BayesianEstimator, t_distribution

import matplotlib.pyplot as plt
import numpy as np

def expected_value(distributions: List[t_distribution]) -> float:
    """
    Liczy 'sredni' rozkład prawdopodobieństwa z przekazaynych wyników estymacji
    """
    e_p_a_09 = np.mean( [distribution[0.9] for distribution in distributions] )
    e_p_a_08 = np.mean( [distribution[0.8] for distribution in distributions] )
    
    return {0.8: e_p_a_08, 0.9: e_p_a_09}

def variance(distributions: List[t_distribution]) -> float:
    """
    Liczy wariancję dla wartoci rozkładu prawdopodobieństwa p(a|Y)
    """
    e_var_09 = np.var( [distribution[0.9] for distribution in distributions] )
    e_var_08 = np.var( [distribution[0.8] for distribution in distributions] )
    
    return {0.8: e_var_08, 0.9: e_var_09}

def test_helper(x_size: float, p0: t_distribution) -> Dict[ float, Dict[float,float] ]:
    """
    Funkcja pomocnicza. Ustala schemat testowy dla różnych rozmiarów wektora
    pomiarów oraz prawdopodobieństw 'a priori'
    
    Parametry:
            x_size  : rozmiar pomiarów
            p0      : rozkład a priori parametru a
    """    
    x0 = 0    
    # dla a=0.8
    a = 0.8
        
    x = [x0]    
    for i in range(x_size):
        x.append(a*x[-1] + np.random.normal(0,1))        
        
    x = np.array(x)
    x = np.resize(x, (x_size,1))
    
    est = BayesianEstimator(p0,x)
    p = { 0.8 : est.estimate()}
        
    # dla a=0.9
    a = 0.9
        
    x = [x0]    
    for i in range(x_size):
        x.append(a*x[-1] + np.random.normal(0,1))        
    x = np.array(x)
    x = np.resize(x, (x_size,1))
    
    est = BayesianEstimator(p0,x)
    p[0.9] = est.estimate()
        
    return {0.8 : p[0.8], 0.9 : p[0.9]}

def test_helper_2(x_size: float, p0: t_distribution, a_change_id: float, a_change_size: float) -> Dict[ float, Dict[float,float] ]:
    """
    Funkcja pomocnicza. Ustala schemat testowy dla różnych rozmiarów wektora
    pomiarów oraz prawdopodobieństw 'a priori'
    
    Parametry:
            x_size          : rozmiar pomiarów
            p0              : rozkład a priori parametru a
            a_change_id     : indeks pomiaru, dla którego zmieniona jest wartosć parametru a
            a_change_size: ilosć kolejnych pomiarów ze zmienioną wartoscią parametru a
    """    
    x0 = 0    
    # dla a=0.8
    a = 0.8
        
    x = [x0]    
    for i in range(a_change_id):
        x.append(a*x[-1] + np.random.normal(0,1))        
    a=0.9
    for _ in range(a_change_size):
        x.append(a*x[-1] + np.random.normal(0,1))
    a=0.8
    for _ in range(x_size - a_change_id - a_change_size):
        x.append(a*x[-1] + np.random.normal(0,1))
        
    x = np.array(x)
    x = np.resize(x, (x_size,1))
    
    est = BayesianEstimator(p0,x)
    p = { 0.8 : est.estimate()}
        
    # dla a=0.9
    a = 0.9
        
    x = [x0]    
    for i in range(a_change_id):
        x.append(a*x[-1] + np.random.normal(0,1))        
    a=0.8
    for _ in range(a_change_size):
        x.append(a*x[-1] + np.random.normal(0,1))
    a=0.9
    for _ in range(x_size - a_change_id - a_change_size):
        x.append(a*x[-1] + np.random.normal(0,1))
        
    x = np.array(x)
    x = np.resize(x, (x_size,1))
    
    est = BayesianEstimator(p0,x)
    p[0.9] = est.estimate()
        
    return {0.8 : p[0.8], 0.9 : p[0.9]}
    
def different_data_length():
    """
    Sprawdza zależnosć pomiędzy długoscią wektora danych, a uzyskanym rozkładem.
    Wykonywanych jest kilka (10) prób dla danej długosci wektora aby uniknąć
    losowosci
    """
    def different_data_length_case_1() -> Dict[ float, Dict[float,float] ]:
        """
        Długosć wektora danych : 50
        """
        x_size = 50
        p0 = {0.8:0.5, 0.9:0.5}
        
        return test_helper(x_size, p0)
        
    def different_data_length_case_2() -> Dict[ float, Dict[float,float] ]:
        """
        Długosć wektora danych : 100
        """
        x_size = 100
        p0 = {0.8:0.5, 0.9:0.5}
        
        return test_helper(x_size, p0)
    
    def different_data_length_case_3() -> Dict[ float, Dict[float,float] ]:
        """
        Długosć wektora danych : 250
        """
        x_size = 250
        p0 = {0.8:0.5, 0.9:0.5}
        
        return test_helper(x_size, p0)
    
    def different_data_length_case_4() -> Dict[ float, Dict[float,float] ]:
        """
        Długosć wektora danych : 500
        """
        x_size = 500
        p0 = {0.8:0.5, 0.9:0.5}
        
        return test_helper(x_size, p0)
    
    def different_data_length_case_5() -> Dict[ float, Dict[float,float] ]:
        """
        Długosć wektora danych : 1000
        """
        x_size = 1000
        p0 = {0.8:0.5, 0.9:0.5}
        
        return test_helper(x_size, p0)
    
    iterations = 40
    estimations = dict()

    estimations['case_1'] = [different_data_length_case_1() for _ in range(iterations)]
    estimations['case_2'] = [different_data_length_case_2() for _ in range(iterations)]
    estimations['case_3'] = [different_data_length_case_3() for _ in range(iterations)]
    estimations['case_4'] = [different_data_length_case_4() for _ in range(iterations)]
    estimations['case_5'] = [different_data_length_case_5() for _ in range(iterations)]
    
    wrong_08s = list()
    wrong_09s = list()
    # dla każdego badanego przypadku
    for case in estimations.values():
        wrong_08 = 0
        wrong_09 = 0
        # dla każdej próby
        for est in case:            
            # gdy a=0.8
            # jeżeli p(a=0.8|X)<p(a=0.9|X) - błędna estymacja
            if est[0.8][0.8] < est[0.8][0.9]:
                wrong_08 += 1
                    
            # gdy a=0.9
            # jeżeli p(a=0.9|X)<p(a=0.8|X) - błędna estymacja
            if est[0.9][0.9] < est[0.9][0.8]:
                wrong_09 += 1
        wrong_08s.append(wrong_08)                
        wrong_09s.append(wrong_09)        
                
        
    print(" ----------------------------------------------------------------------------------------------------- ")
    print(" --- Test wpływu ilości próbek na wynik estymacji - przypadek dla a=0.8 --- ")
    plt.figure(1)
    plt.plot( [50 for _ in estimations['case_1']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_1'] ], '*')
    plt.gca().set_prop_cycle(None)
    print('Wartość oczekiwana z wartości rozkładów dla 50 próbek: ', expected_value( [est[0.8] for est in estimations['case_1']] ))
    print('Wariancja z wartości rozkładów dla 50 próbek: ', variance([est[0.8] for est in estimations['case_1']]))
    print('Liczba błędnych estymacji dla 50 próbek (na ', iterations, ' iteracji ): ', wrong_08s[0], '\n')
    plt.plot( [100 for _ in estimations['case_2']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_2'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Wartość oczekiwana z wartości rozkładów dla 100 próbek: ', expected_value([est[0.8] for est in estimations['case_2']]))
    print('Wariancja z wartości rozkładów dla 100 próbek: ', variance([est[0.8] for est in estimations['case_2']]))
    print('Liczba błędnych estymacji dla 100 próbek (na ', iterations, ' iteracji ): ', wrong_08s[1], '\n')
    plt.plot( [250 for _ in estimations['case_3']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_3'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Wartość oczekiwana z wartości rozkładów dla 250 próbek: ', expected_value([est[0.8] for est in estimations['case_3']]))
    print('Wariancja z wartości rozkładów dla 250 próbek: ', variance([est[0.8] for est in estimations['case_3']]))
    print('Liczba błędnych estymacji dla 250 próbek (na ', iterations, ' iteracji ): ', wrong_08s[2], '\n')
    plt.plot( [500 for _ in estimations['case_4']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_4'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Wartość oczekiwana z wartości rozkładów dla 500 próbek: ', expected_value([est[0.8] for est in estimations['case_4']]))
    print('Wariancja z wartości rozkładów dla 500 próbek: ', variance([est[0.8] for est in estimations['case_4']]))
    print('Liczba błędnych estymacji dla 500 próbek (na ', iterations, ' iteracji ): ', wrong_08s[3], '\n')
    plt.plot( [1000 for _ in estimations['case_5']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_5'] ], '*' )
    print('Wartość oczekiwana z wartości rozkładów dla 1000 próbek: ', expected_value([est[0.8] for est in estimations['case_5']]))
    print('Wariancja z wartości rozkładów dla 1000 próbek: ', variance([est[0.8] for est in estimations['case_5']]))
    print('Liczba błędnych estymacji dla 1000 próbek (na ', iterations, ' iteracji ): ', wrong_08s[4], '\n')
    plt.legend(['p(a=0.8|X)','p(a=0.9|X)'])
    plt.title('Rozkład estymatora dla różnych wielkości \n zbioru danych (przypadek dla a=0.8)')
    plt.xlabel('Wielkość wektora danych')
    plt.ylabel(r'$p(a = a_i | X)$')
    plt.savefig('1.png',dpi=300)
    
    print(" --- Test wpływu ilości próbek na wynik estymacji - przypadek dla a=0.9 --- ")
    plt.figure(2)
    plt.plot( [50 for _ in estimations['case_1']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_1'] ], '*')
    plt.gca().set_prop_cycle(None)
    print('Wartość oczekiwana z wartości rozkładów dla 50 próbek: ', expected_value( [est[0.9] for est in estimations['case_1']] ))
    print('Wariancja z wartości rozkładów dla 50 próbek: ', variance([est[0.9] for est in estimations['case_1']]))
    print('Liczba błędnych estymacji dla 50 próbek (na ', iterations, ' iteracji ): ', wrong_09s[0], '\n')
    plt.plot( [100 for _ in estimations['case_2']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_2'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Wartość oczekiwana z wartości rozkładów dla 100 próbek: ', expected_value([est[0.9] for est in estimations['case_2']]))
    print('Wariancja z wartości rozkładów dla 100 próbek: ', variance([est[0.9] for est in estimations['case_2']]))
    print('Liczba błędnych estymacji dla 100 próbek (na ', iterations, ' iteracji ): ', wrong_09s[1])
    plt.plot( [250 for _ in estimations['case_3']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_3'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Wartość oczekiwana z wartości rozkładów dla 250 próbek: ', expected_value([est[0.9] for est in estimations['case_3']]))
    print('Wariancja z wartości rozkładów dla 250 próbek: ', variance([est[0.9] for est in estimations['case_3']]))
    print('Liczba błędnych estymacji dla 250 próbek (na ', iterations, ' iteracji ): ', wrong_09s[2], '\n')
    plt.plot( [500 for _ in estimations['case_4']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_4'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Wartość oczekiwana z wartości rozkładów dla 500 próbek: ', expected_value([est[0.9] for est in estimations['case_4']]))
    print('Wariancja z wartości rozkładów dla 500 próbek: ', variance([est[0.9] for est in estimations['case_4']]))
    print('Liczba błędnych estymacji dla 500 próbek (na ', iterations, ' iteracji ): ', wrong_09s[3], '\n')
    plt.plot( [1000 for _ in estimations['case_5']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_5'] ], '*' )
    print('Wartość oczekiwana z wartości rozkładów dla 1000 próbek: ', expected_value([est[0.9] for est in estimations['case_5']]))
    print('Wariancja z wartości rozkładów dla 1000 próbek: ', variance([est[0.9] for est in estimations['case_5']]))
    print('Liczba błędnych estymacji dla 1000 próbek (na ', iterations, ' iteracji ): ', wrong_09s[4], '\n')
    plt.legend(['p(a=0.8|X)','p(a=0.9|X)'])
    plt.title('Rozkład estymatora dla różnych wielkości \n zbioru danych (przypadek dla a=0.9)')
    plt.xlabel('Wielkość wektora danych')
    plt.ylabel(r'$p(a = a_i | X)$')
    plt.savefig('2.png',dpi=300)
    
    return estimations

def different_priori_prob():
    """
    Testuje sprawnosć estmatora pod względem czułosci na zaproponowany 
    rozkład a priori parametru 'a'. Wykonywanych jest kilka (10) prób dla 
    danego rozkładu aby uniknąć losowosci.
    """
    def different_priori_prob_case_1():
        """
        p0 = {0.8:0.9, 0.9:0.1}
        """
        x_size = 250
        p0 = {0.8:0.9, 0.9:0.1}
        
        return test_helper(x_size, p0)
    
    def different_priori_prob_case_2():
        """
        p0 = {0.8:0.25, 0.9:0.75}
        """
        x_size = 250
        p0 = {0.8:0.25, 0.9:0.75}
        
        return test_helper(x_size, p0)
    
    def different_priori_prob_case_3():
        """
        p0 = {0.8:0.5, 0.9:0.5}
        """
        x_size = 250
        p0 = {0.8:0.5, 0.9:0.5}
        
        return test_helper(x_size, p0)
    
    def different_priori_prob_case_4():
        """
        p0 = {0.8:0.25, 0.9:0.75}
        """
        x_size = 250
        p0 = {0.8:0.25, 0.9:0.75}
        
        return test_helper(x_size, p0)
    
    def different_priori_prob_case_5():
        """
        p0 = {0.8:0.1, 0.9:0.9}
        """
        x_size = 250
        p0 = {0.8:0.1, 0.9:0.9}
        
        return test_helper(x_size, p0)
    
    iterations = 40
    estimations = dict()

    estimations['case_1'] = [different_priori_prob_case_1() for _ in range(iterations)]
    estimations['case_2'] = [different_priori_prob_case_2() for _ in range(iterations)]
    estimations['case_3'] = [different_priori_prob_case_3() for _ in range(iterations)]
    estimations['case_4'] = [different_priori_prob_case_4() for _ in range(iterations)]
    estimations['case_5'] = [different_priori_prob_case_5() for _ in range(iterations)]
    
    wrong_08s = list()
    wrong_09s = list()
    # dla każdego badanego przypadku
    for case in estimations.values():
        wrong_08 = 0
        wrong_09 = 0
        # dla każdej próby
        for est in case:            
            # gdy a=0.8
            # jeżeli p(a=0.8|X)<p(a=0.9|X) - błędna estymacja
            if est[0.8][0.8] < est[0.8][0.9]:
                wrong_08 += 1
                    
            # gdy a=0.9
            # jeżeli p(a=0.9|X)<p(a=0.8|X) - błędna estymacja
            if est[0.9][0.9] < est[0.9][0.8]:
                wrong_09 += 1
        wrong_08s.append(wrong_08)                
        wrong_09s.append(wrong_09)    
        
    x = np.array([1,2,3,4,5])
    
    print(" ----------------------------------------------------------------------------------------------------- ")
    print(" --- Test wpływu założonego rozkładu \'a priori\' na wynik estymacji - przypadek dla a=0.8 --- ")
    plt.figure(3)
    my_xticks = ['{0.8:0.9, 0.9:0.1}','{0.8:0.75, 0.9:0.25}','{0.8:0.5, 0.9:0.5}', '{0.8:0.25, 0.9:0.75}','{0.8:0.1, 0.9:0.9}']
    plt.xticks(x, my_xticks, fontsize=7)
    plt.plot( [1 for _ in estimations['case_1']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_1'] ], '*')
    plt.gca().set_prop_cycle(None)
    print('Dla rozkładu \'a priori\' {0.8:0.9, 0.9:0.1}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.8] for est in estimations['case_1']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.8] for est in estimations['case_1']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_08s[0], '\n')
    plt.plot( [2 for _ in estimations['case_2']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_2'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla rozkładu \'a priori\' {0.8:0.75, 0.9:0.25}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.8] for est in estimations['case_2']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.8] for est in estimations['case_2']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_08s[1], '\n')
    plt.plot( [3 for _ in estimations['case_3']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_3'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla rozkładu \'a priori\' {0.8:0.5, 0.9:0.5}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.8] for est in estimations['case_3']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.8] for est in estimations['case_3']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_08s[2], '\n')
    plt.plot( [4 for _ in estimations['case_4']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_4'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla rozkładu \'a priori\' {0.8:0.25, 0.9:0.75}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.8] for est in estimations['case_4']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.8] for est in estimations['case_4']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_08s[3], '\n')
    plt.plot( [5 for _ in estimations['case_5']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_5'] ], '*' )
    print('Dla rozkładu \'a priori\' {0.8:0.1, 0.9:0.9}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.8] for est in estimations['case_5']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.8] for est in estimations['case_5']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_08s[4], '\n')
    plt.legend(['p(a=0.8|X)','p(a=0.9|X)'])
    plt.title('Rozkład estymatora dla różnych \n prawdopodobieństw \'a priori\' (przypadek dla a=0.8)')
    plt.xlabel('Prawdopodobieństwa \' a priori \'')
    plt.ylabel(r'$p(a = a_i | X)$')
    plt.savefig('3.png',dpi=300)
    
    print(" --- Test wpływu założonego rozkładu \'a priori\' na wynik estymacji - przypadek dla a=0.9 --- ")
    plt.figure(4)
    my_xticks = ['{0.8:0.9, 0.9:0.1}','{0.8:0.75, 0.9:0.25}','{0.8:0.5, 0.9:0.5}', '{0.8:0.25, 0.9:0.75}','{0.8:0.1, 0.9:0.9}']
    plt.xticks(x, my_xticks, fontsize=7)
    plt.plot( [1 for _ in estimations['case_1']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_1'] ], '*')
    plt.gca().set_prop_cycle(None)
    print('Dla rozkładu \'a priori\' {0.8:0.9, 0.9:0.1}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.9] for est in estimations['case_1']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.9] for est in estimations['case_1']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_09s[0], '\n')
    plt.plot( [2 for _ in estimations['case_2']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_2'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla rozkładu \'a priori\' {0.8:0.75, 0.9:0.25}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.9] for est in estimations['case_2']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.9] for est in estimations['case_2']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_09s[1], '\n')
    plt.plot( [3 for _ in estimations['case_3']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_3'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla rozkładu \'a priori\' {0.8:0.5, 0.9:0.5}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.9] for est in estimations['case_3']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.9] for est in estimations['case_3']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_09s[2], '\n')
    plt.plot( [4 for _ in estimations['case_4']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_4'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla rozkładu \'a priori\' {0.8:0.25, 0.9:0.75}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.9] for est in estimations['case_4']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.9] for est in estimations['case_4']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_09s[3], '\n')
    plt.plot( [5 for _ in estimations['case_5']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_5'] ], '*' )
    print('Dla rozkładu \'a priori\' {0.8:0.1, 0.9:0.9}')
    print('Wartość oczekiwana z wartości rozkładów: ', expected_value( [est[0.9] for est in estimations['case_5']] ))
    print('Wariancja z wartości rozkładów: ', variance([est[0.9] for est in estimations['case_5']]))
    print('Liczba błędnych estymacji:  (na ', iterations, ' iteracji ): ', wrong_09s[4], '\n')
    plt.legend(['p(a=0.8|X)','p(a=0.9|X)'])
    plt.title('Rozkład estymatora dla różnych \n prawdopodobieństw \'a priori\' (przypadek dla a=0.9)')
    plt.xlabel('Prawdopodobieństwa \' a priori \'')
    plt.ylabel(r'$p(a = a_i | X)$')
    plt.savefig('4.png',dpi=300)
    
def parameter_change_during_test():
    
    def parameter_change_during_test_case_1():
        x_size = 500
        p0 = {0.8: 0.5, 0.9:0.5}
        a_change_id = 350
        a_change_size = 100
        
        return test_helper_2(x_size, p0, a_change_id, a_change_size)
    
    def parameter_change_during_test_case_2():
        x_size = 500
        p0 = {0.8: 0.5, 0.9:0.5}
        a_change_id = 0
        a_change_size = 100
        
        return test_helper_2(x_size, p0, a_change_id, a_change_size)
    
    def parameter_change_during_test_case_3():
        x_size = 500
        p0 = {0.8: 0.5, 0.9:0.5}
        a_change_id = 350
        a_change_size = 10
        
        return test_helper_2(x_size, p0, a_change_id, a_change_size)
    
    def parameter_change_during_test_case_4():
        x_size = 500
        p0 = {0.8: 0.5, 0.9:0.5}
        a_change_id = 0
        a_change_size = 10
        
        return test_helper_2(x_size, p0, a_change_id, a_change_size)
    
    
    iterations = 40
    estimations = dict()

    estimations['case_1'] = [parameter_change_during_test_case_1() for _ in range(iterations)]
    estimations['case_2'] = [parameter_change_during_test_case_2() for _ in range(iterations)]
    estimations['case_3'] = [parameter_change_during_test_case_3() for _ in range(iterations)]
    estimations['case_4'] = [parameter_change_during_test_case_4() for _ in range(iterations)]
    
    wrong_08s = list()
    wrong_09s = list()
    # dla każdego badanego przypadku
    for case in estimations.values():
        wrong_08 = 0
        wrong_09 = 0
        # dla każdej próby
        for est in case:            
            # gdy a=0.8
            # jeżeli p(a=0.8|X)<p(a=0.9|X) - błędna estymacja
            if est[0.8][0.8] < est[0.8][0.9]:
                wrong_08 += 1
                    
            # gdy a=0.9
            # jeżeli p(a=0.9|X)<p(a=0.8|X) - błędna estymacja
            if est[0.9][0.9] < est[0.9][0.8]:
                wrong_09 += 1
        wrong_08s.append(wrong_08)                
        wrong_09s.append(wrong_09)   
        
    x = np.array([1,2,3,4,5])
    
    print(" ----------------------------------------------------------------------------------------------------- ")
    print(" --- Test wpływu zmian parametru a w czasie pomiaru - przypadek dla a=0.8 (chwilowa zmiana na 0.9) --- ")
    plt.figure(5)
    my_xticks = ['[350,100]','[0,100]','[350,10]','[0,10]']
    plt.xticks(x, my_xticks, fontsize=7)
    plt.plot( [1 for _ in estimations['case_1']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_1'] ], '*')
    plt.gca().set_prop_cycle(None)
    print('Dla zaburzenia [350,100]')
    print('Wartość oczekiwana z wartości rozkładów dla zaburzenia a=0.8: ', expected_value( [est[0.8] for est in estimations['case_1']] ))
    print('Wariancja z wartości rozkładów dla zaburzenia a=0.8: ', variance([est[0.8] for est in estimations['case_1']]))
    print('Liczba błędnych estymacji dla zaburzenia a=0.8:  (na ', iterations, ' iteracji ): ', wrong_08s[0], '\n')
    plt.plot( [2 for _ in estimations['case_2']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_2'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla zaburzenia [0,100]')
    print('Wartość oczekiwana z wartości rozkładów dla zaburzenia a=0.8: ', expected_value( [est[0.8] for est in estimations['case_2']] ))
    print('Wariancja z wartości rozkładów dla zaburzenia a=0.8: ', variance([est[0.8] for est in estimations['case_2']]))
    print('Liczba błędnych estymacji dla zaburzenia a=0.8:  (na ', iterations, ' iteracji ): ', wrong_08s[1], '\n')
    plt.plot( [3 for _ in estimations['case_3']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_3'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla zaburzenia [350,10]')
    print('Wartość oczekiwana z wartości rozkładów dla zaburzenia a=0.8: ', expected_value( [est[0.8] for est in estimations['case_3']] ))
    print('Wariancja z wartości rozkładów dla zaburzenia a=0.8: ', variance([est[0.8] for est in estimations['case_3']]))
    print('Liczba błędnych estymacji dla zaburzenia a=0.8:  (na ', iterations, ' iteracji ): ', wrong_08s[2], '\n')
    plt.plot( [4 for _ in estimations['case_4']], [ np.array([v for v in est[0.8].values()]) for est in estimations['case_4'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla zaburzenia [0,10]')
    print('Wartość oczekiwana z wartości rozkładów dla zaburzenia a=0.8: ', expected_value( [est[0.8] for est in estimations['case_4']] ))
    print('Wariancja z wartości rozkładów dla zaburzenia a=0.8: ', variance([est[0.8] for est in estimations['case_4']]))
    print('Liczba błędnych estymacji dla zaburzenia a=0.8:  (na ', iterations, ' iteracji ): ', wrong_08s[3], '\n')
    plt.legend(['p(a=0.8|X)','p(a=0.9|X)'])
    plt.title('Rozkład estymatora dla zaburzenia a=0.8')
    plt.xlabel('Rodzaj zaburzenia [indeks pierwszego zaburzenia, długość zaburzenia]')
    plt.ylabel(r'$p(a = a_i | X)$')
    plt.savefig('5.png',dpi=300)
    
    print(" --- Test wpływu zmian parametru a w czasie pomiaru - przypadek dla a=0.9 (chwilowa zmiana na 0.8) --- ")
    plt.figure(6)
    my_xticks = ['[350,100]','[0,100]','[350,10]','[0,10]']
    plt.xticks(x, my_xticks, fontsize=7)
    plt.plot( [1 for _ in estimations['case_1']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_1'] ], '*')
    plt.gca().set_prop_cycle(None)
    print('Dla zaburzenia [350,100]')
    print('Wartość oczekiwana z wartości rozkładów dla zaburzenia a=0.9: ', expected_value( [est[0.9] for est in estimations['case_1']] ))
    print('Wariancja z wartości rozkładów dla zaburzenia a=0.9: ', variance([est[0.9] for est in estimations['case_1']]))
    print('Liczba błędnych estymacji dla zaburzenia a=0.9:  (na ', iterations, ' iteracji ): ', wrong_09s[0], '\n')
    plt.plot( [2 for _ in estimations['case_2']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_2'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla zaburzenia [0,100]')
    print('Wartość oczekiwana z wartości rozkładów dla zaburzenia a=0.9: ', expected_value( [est[0.9] for est in estimations['case_2']] ))
    print('Wariancja z wartości rozkładów dla zaburzenia a=0.9: ', variance([est[0.9] for est in estimations['case_2']]))
    print('Liczba błędnych estymacji dla zaburzenia a=0.9:  (na ', iterations, ' iteracji ): ', wrong_09s[1], '\n')
    plt.plot( [3 for _ in estimations['case_3']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_3'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla zaburzenia [350,10]')
    print('Wartość oczekiwana z wartości rozkładów dla zaburzenia a=0.9: ', expected_value( [est[0.9] for est in estimations['case_3']] ))
    print('Wariancja z wartości rozkładów dla zaburzenia a=0.9: ', variance([est[0.9] for est in estimations['case_3']]))
    print('Liczba błędnych estymacji dla zaburzenia a=0.9:  (na ', iterations, ' iteracji ): ', wrong_09s[2], '\n')
    plt.plot( [4 for _ in estimations['case_4']], [ np.array([v for v in est[0.9].values()]) for est in estimations['case_4'] ], '*' )
    plt.gca().set_prop_cycle(None)
    print('Dla zaburzenia [0,10]')
    print('Wartość oczekiwana z wartości rozkładów dla zaburzenia a=0.9: ', expected_value( [est[0.9] for est in estimations['case_4']] ))
    print('Wariancja z wartości rozkładów dla zaburzenia a=0.9: ', variance([est[0.9] for est in estimations['case_4']]))
    print('Liczba błędnych estymacji dla zaburzenia a=0.9:  (na ', iterations, ' iteracji ): ', wrong_09s[3], '\n')
    plt.legend(['p(a=0.8|X)','p(a=0.9|X)'])
    plt.title('Rozkład estymatora dla zaburzenia a=0.9')
    plt.xlabel('Rodzaj zaburzenia [indeks pierwszego zaburzenia, długość zaburzenia]')
    plt.ylabel(r'$p(a = a_i | X)$')
    plt.savefig('6.png',dpi=300)
        
    
if __name__ == "__main__":
#    import sys 

#    stdoutOrigin=sys.stdout 
#    sys.stdout = open("log.txt", "w")
    
    different_data_length()
    different_priori_prob()
    parameter_change_during_test()
    
#    sys.stdout.close()
#    sys.stdout=stdoutOrigin