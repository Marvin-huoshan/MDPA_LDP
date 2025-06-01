import numpy as np
import random
from tqdm import tqdm


def GRR(X, ratio, domain, epsilon, n, target_set, splits):
    '''
    Perform the GRR protocol
    :param X: The real values for each users
    :param ratio: fake users ratio
    :param domain: domain size
    :param epsilon: privacy budget
    :param n: number of users
    :param target_set: fake users target set
    :return: support_list, one_list, ESTIMATE_DIST, ESTIMATE_Input
    '''
    Report = np.zeros(n, dtype=int)
    p = np.exp(epsilon) / (np.exp(epsilon) + domain - 1)
    q = 1.0 / (np.exp(epsilon) + domain - 1)
    # random.shuffle(X)
    for i in tqdm(range(n)):
        if i < n * (1 - ratio):
            v = X[i]
            # intdigest -> get integer value from xxhash(Output of xxhash is a hash item)
            p_sample = np.random.random_sample()
            if p_sample < p:
                y = v
            else:
                y = np.random.choice([k for k in range(domain) if k != v])
            Report[i] = y
        else:
            splits_list = random.sample(list(target_set), splits)
            y = random.choice(splits_list)
            Report[i] = y
    est_dist = np.bincount(Report, minlength=domain)
    est_dist = (est_dist - n * q) / (p - q)
    return est_dist
