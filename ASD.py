import numpy as np
import math
from sympy import *
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.optimize import minimize

def ASD(est_dist, n, epsilon, domain, perturb_method):
    '''
    Overall detection method ASD
    :param est_dist: estimation
    :param n: number of users
    :param epsilon: privacy budget
    :param domain: domain size
    :param perturb_method: LDP protocols
    :return: Whether there is an attack
    '''
    lp, lph, zero_std = lower_point(n, epsilon, domain, est_dist, perturb_method)
    estimates = np.copy(est_dist)
    flag = 0
    estimates = norms(estimates, n)
    print('zero_std: ', zero_std)
    confidence_level, n_percent, z_score = find_confidence_level(n, domain, zero_std, estimates)
    valid_indices = [idx for idx, value in enumerate(estimates) if z_score * zero_std < value]
    valid_values = [estimates[idx] for idx in valid_indices]
    sum_valid_values = sum(valid_values)
    print('n_percent: ', n_percent)
    print('sum: ', sum_valid_values)
    if sum_valid_values > n:
        print('Attacking!')
        flag = 1
        return flag
    return flag

def norms(est, n):
    '''
    ensure sum to n
    :param est: estimation
    :param n: number of users
    :return: norms estimation
    '''
    while(np.fabs(sum(est) - n) > 1):
        sum_estimate = np.sum(est)
        print(sum_estimate)
        diff_pre = (n - sum_estimate) / len(est)
        est += diff_pre
        est = [0 if x > n else x for x in est]
    est = np.array(est)
    return est

def lower_point(n, epsilon, domain, est_dist, perturb_method):
    '''
    count the lower point at the estimation to split this dist
    :param est_dist: estimation
    :param n: number of users
    :param epsilon: privacy budget
    :return:
    '''
    if perturb_method == 'OUE':
        q_OUE = 1 / (math.exp(epsilon) + 1)
        p = 1 / 2
        q = q_OUE
    elif perturb_method in ('OLH_User', 'OLH_Server'):
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1 / g
    elif perturb_method in ('HST_Server', 'HST_User'):
        g = 2
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1 / g
    elif perturb_method == 'GRR':
        p = np.exp(epsilon) / (np.exp(epsilon) + domain - 1)
        q = 1.0 / (np.exp(epsilon) + domain - 1)
    else:
        p = 0
        q = 0
    r_h = np.max(est_dist) / sum(est_dist)
    max_std = sqrt(n*q*(1-q))
    max_std = max_std / (p - q)
    x = symbols('x')
    y = solve(2 * sqrt(n*(x * p * (1 - p) + (1 - x) * q * (1 - q)))
              - n * (p - q) * x, x)
    y1 = y.pop()
    y_high = 3 * sqrt(n * q * (1 - q)) / (p - q)
    y3 = y_high
    y2 = (1 * sqrt(n*(y1 * p * (1 - p) + (1 - y1) * q * (1 - q)))
          + n * (p - q) * y1) / (p - q)
    y4 = (1 * sqrt(n*(y1 * p * (1 - p) + (1 - y1) * q * (1 - q))) / (p - q)
          + n * y1)
    return y1, y4, max_std

def find_confidence_level(n, d, s, est, confidence_start=0.90, confidence_end=0.9999, step=0.0001):
    '''
    Find the minimum confidence level that satisfies the given inequality.
    :param n: number of users
    :param d: domain size
    :param s: maximum standard deviation
    :param est: estimation
    :param confidence_start: starting confidence level
    :param confidence_end: ending confidence level
    :param step: Confidence traversal step
    :return: the first Confidence that satisfied the condition
    '''
    lambdas = 0.02
    n_percent = lambdas * n
    for confidence in np.arange(confidence_start, confidence_end + step, step):
        z_score = norm.ppf((1 + confidence) / 2)
        x = z_score * s
        est_min = np.min(est)
        l = len([num for num in est if num < abs(est_min)])
        lhs = x * l * (1 - confidence) * 0.5
        if lhs < n_percent:
            return confidence, lambdas, z_score
    return None, None, 3.8906
