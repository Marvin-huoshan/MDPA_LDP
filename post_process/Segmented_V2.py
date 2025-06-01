import numpy as np
import math
from sympy import *


def Segmented_V2(est_dist, n, epsilon, perturb_method):
    if perturb_method == 'OUE':
        _, lph, _ = lower_point_OUE(n, epsilon, est_dist)
    elif perturb_method in ('OLH_User', 'OLH_Server'):
        _, lph, _ = lower_point_OLH(n, epsilon, est_dist)
    elif perturb_method in ('HST_Server', 'HST_User'):
        _, lph, _ = lower_point_HST(n, epsilon, est_dist)
    elif perturb_method == 'GRR':
        _, lph, _ = lower_point_GRR(n, epsilon, est_dist, len(est_dist))
    estimates = np.copy(est_dist)
    sum_estimate = np.sum(estimates)
    #print(sum_estimate)
    total = sum_estimate
    diff_pre = (n - sum_estimate) / len(estimates)
    estimates += diff_pre
    while (estimates < 0).any():
        sum_neg = np.sum(estimates[estimates < 0])
        #print(sum_neg)
        total_before = sum(estimates)
        estimates[estimates < 0] = 0
        total_after = sum(estimates)
        #mask = estimates > 0
        mask1 = (0 < estimates) & (estimates < lph)
        diff1 = (total_before - total_after) / sum(mask1)
        estimates[mask1] += diff1
        total = sum(estimates)
    #print('total: ', total)
    return estimates * n / total

def lower_point_OLH(n, epsilon, est_dist):
    '''
    count the lower point at the estimation to split this dist
    :param est_dist:
    :param n:
    :param epsilon:
    :return:
    '''
    r_h = np.max(est_dist) / sum(est_dist)
    g = int(round(math.exp(epsilon))) + 1
    p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
    q = 1 / g
    x = symbols('x')
    y = solve(2 * sqrt(n*(x * p * (1 - p) + (1 - x) * q * (1 - q)))- n * (p - q) * x, x)
    y1 = y.pop()
    y_high = 3 * sqrt(n * q * (1 - q)) / (p - q)
    y3 = y_high
    y2 = (2 * sqrt(n*(y1 * p * (1 - p) + (1 - y1) * q * (1 - q))) + n * (p - q) * y1) / (p - q)
    return y1, y2, y3

def lower_point_OUE(n, epsilon, est_dist):
    '''                                                                                                                  
    count the lower point at the estimation to split this dist                                                           
    :param est_dist:                                                                                                     
    :param n:                                                                                                            
    :param epsilon:                                                                                                      
    :return:                                                                                                             
    '''
    r_h = np.max(est_dist) / sum(est_dist)
    q = 1 / (math.exp(epsilon) + 1)
    p = 0.5
    x = symbols('x')
    y = solve(2 * sqrt(n*(x * p * (1 - p) + (1 - x) * q * (1 - q))) - n * (p - q) * x, x)
    y1 = y.pop()
    y_high = 3 * sqrt(n * q * (1 - q)) / (p - q)
    y3 = y_high
    y2 = (2 * sqrt(n*(y1 * p * (1 - p) + (1 - y1) * q * (1 - q))) + n * (p - q) * y1) / (p - q)
    return y1, y2, y3

def lower_point_HST(n, epsilon, est_dist):
    '''
    count the lower point at the estimation to split this dist
    :param est_dist:
    :param n:
    :param epsilon:
    :return:
    '''
    r_h = np.max(est_dist) / sum(est_dist)
    g = 2
    p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
    q = 1 / g
    x = symbols('x')
    y = solve(2 * sqrt(n*(x * p * (1 - p) + (1 - x) * q * (1 - q)))- n * (p - q) * x, x)
    y1 = y.pop()
    y_high = 3 * sqrt(n * q * (1 - q)) / (p - q)
    y3 = y_high
    y2 = (2 * sqrt(n*(y1 * p * (1 - p) + (1 - y1) * q * (1 - q))) + n * (p - q) * y1) / (p - q)
    return y1, y2, y3

def lower_point_GRR(n, epsilon, est_dist, d):
    '''
    count the lower point at the estimation to split this dist
    :param est_dist:
    :param n:
    :param epsilon:
    :return:
    '''
    p = math.exp(epsilon) / (math.exp(epsilon) + d - 1)
    q = 1 / (math.exp(epsilon) + d - 1)
    x = symbols('x')
    y = solve(2 * sqrt(n*(x * p * (1 - p) + (1 - x) * q * (1 - q)))- n * (p - q) * x, x)
    y1 = y.pop()
    y_high = 3 * sqrt(n * q * (1 - q)) / (p - q)
    y3 = y_high
    y2 = (2 * sqrt(n*(y1 * p * (1 - p) + (1 - y1) * q * (1 - q))) + n * (p - q) * y1) / (p - q)
    return y1, y2, y3
