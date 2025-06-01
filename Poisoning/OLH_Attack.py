import random
from multiprocessing import Pool
import numpy as np
import xxhash


def hash_1000(target_set, g, domain, untarget_map):
    '''
    Each fake user randomly sample 1000 hash functions and use one that hashes the most
    target items to the same value.
    :param target_set: target items set.
    :param g: hash function projection domain
    :param domain: domain
    :param untarget_map: how many item should hash map in untarget domain
    :return: The best hash function
    '''
    # select 1000 different hash function
    seed_list = random.sample(range(1,1000000),1000)
    # log the max projection number
    projection_total = 0
    # log the best projection seed
    best_seed = 0
    # log the max projection hash value
    hash_value_with_max_target = 0

    for seeds in seed_list:
        # for each target item
        hash_projection_list = np.zeros(g)
        for item in target_set:
            hash_result = xxhash.xxh32(str(item), seed=seeds).intdigest() % g
            hash_projection_list[hash_result] += 1
        # value target set mapped
        current_max_target_mapped = np.max(hash_projection_list)
        # log all the index of max
        max_indices = np.where(hash_projection_list == current_max_target_mapped)[0]
        hash_value_with_current_max_target = random.choice(max_indices)

        if current_max_target_mapped > projection_total:
            best_seed = seeds
            projection_total = current_max_target_mapped
            hash_value_with_max_target = hash_value_with_current_max_target

    '''print('target_mapped: ', projection_total)
    print('best_seed: ', best_seed)
    print('hash_value: ', hash_value_with_max_target)'''
    return projection_total, best_seed, hash_value_with_max_target

def optimal_proxy_hash(target_set, g):
    '''
    To avoid the hash function map the value not included in the target set same to the target value
    we use one ldeal matrix to proxy the hash function.
    :param target_set: target items set.
    :param g: hash function projection domain
    :return: The best hash function
    '''
    # select 1000 different hash function
    seed_list = random.sample(range(1, 10000), 1000)
    # log the max projection number
    projection_total = 0
    # log the max projection seed
    max_seed = 0
    # log the max projection value
    max_projection_value = 0
    # log the Max projection
    max_projection_value = 0
    for seeds in seed_list:
        # for each target item
        hash_projection_list = np.zeros(g)
        for item in target_set:
            hash_result = xxhash.xxh32(str(item), seed=seeds).intdigest() % g
            hash_projection_list[hash_result] += 1
        if max(hash_projection_list) > projection_total:
            projection_total = np.max(hash_projection_list)
            max_seed = seeds
            max_projection_value = np.argmax(hash_projection_list)
    return projection_total, max_seed, max_projection_value

def Attack_Gain(ECF, f_T, g, m, n, p, q):
    '''
    Measure the Gain from the attack
    :param ECF: Expectation of the characteristic function
    :param f_T: The true frequency of target item
    :param g: number of target items
    :return: Gain from the Attack
    '''
    Gain = (ECF - m * (f_T * (p - q) + g * q)) / ((n + m) * (p - q))
    return Gain
