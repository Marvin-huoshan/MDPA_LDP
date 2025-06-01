import numpy as np
from .processor import Processor
from scipy.stats import norm
import math

def base_cut(est_dist, n, epsilon, d):
    estimates = np.copy(est_dist)
    g = int(round(math.exp(epsilon))) + 1
    p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
    q = 1 / g
    std = math.sqrt(q * (1 - q) / (n * (p - q)**2))
    T = norm.ppf(1 - 2 / d)
    T *= std
    #print('Base cut T: ', T)
    estimates[estimates < T] = 0
    return estimates
