import numpy as np
import xxhash
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import random
# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from functools import partial
from multiprocessing import Pool
import math
from Poisoning.APA import construct_omega

num_samples = 1000000

def process_attacker(i, n, ratio, target_set, g, domain, splits, h_ao, e, K_values, K_probs):
    '''
    Function to process each attacker in parallel.
    Each attacker finds their optimal hash function.
    :param i: Index of the attacker.
    :param n: Total number of users.
    :param ratio: Ratio of attackers to total users.
    :param target_set: Set of target items.
    :param g: Range of hash function outputs (modulo value).
    :param domain: Total domain size.
    :param splits: Number of target items each attacker considers.
    :return: Tuple of (index in User_Seed, best_seed found).
    '''
    # Seed the random number generator uniquely for each process
    k = np.random.choice(K_values, p=K_probs)
    random.seed()
    averge_project_hash = int(domain / g)
    if splits < averge_project_hash:
        # Split the target set for each user
        splits_list = random.sample(list(target_set), splits)
        # Gap between average mapping
        num_map = averge_project_hash
        # Remaining set (unused in this snippet but kept for completeness)
        remaining_set = set(range(domain)) - set(target_set)
        # Adaptive gap between average mapping
        #h_ao = 0
        num_map_AO = random.randint(num_map - h_ao, num_map + h_ao)
        # theoretical APA use the omega_list to replace the num_map_AO num_map_AO = np.random.choice([i for i in range(domain)], construct_omega(e, domain, 'OLH_User'))
        # num_map_AO = np.random.choice([i for i in range(domain)], construct_omega(e, domain, 'OLH_User'))
        non_target_ones = num_map_AO - k
        # Each attacker finds their optimal hash function
        '''best_vector, target_map, diff  = uniform_sampling_best_vector(
            splits_list, g, domain, num_map_AO, num_samples)'''
        target_indices = np.random.choice(list(splits_list), size=k, replace=False)
        non_target_indices = list(set(range(domain)) - set(splits_list))
        non_target_selected = np.random.choice(non_target_indices, size=non_target_ones, replace=False)
        vector = np.zeros(domain, dtype=int)
        vector[target_indices] = 1
        vector[non_target_selected] = 1
    else:
        print('splits > averge_project_hash')
        exit(0)
    # Calculate the index in User_Seed to update
    index = int(n * (1 - ratio) + i)
    print(f'attacker:{i}, target_map:{k}, diff:{num_map_AO - sum(vector)}, h_ao:{h_ao}, splits:{splits}')
    return index, vector

def find_hash_function(seed_list, target_set, domain_eliminate, g, num_map_AO):
    '''
    Find an optimal seed in the seed list that maps as many maps as possible on the target set, with the number of maps as close as possible to num_map_AO on the entire domain domain.
    :param seed_list: seed_list
    :param target_set: target item set
    :param domain_eliminate: untarget itemset
    :param g: hash function domain size
    :param num_map_AO: designed number of support
    :return:
    '''
    # log the max projection number
    best_score = -np.inf
    # log the best projection seed
    best_seed = -1
    # log the target mapped
    best_target_mapped = None
    # log the best hash value
    best_hash_value = None
    # log the min gap
    best_gap = None
    for seed in seed_list:
        hash_projection_list = np.zeros(g)
        hash_other_projection_list = np.zeros(g)
        hash_result = None
        for item in target_set:
            hash_result = xxhash.xxh3_64(str(item), seed=seed).intdigest() % g
            hash_projection_list[hash_result] += 1
        for item in domain_eliminate:
            hash_result = xxhash.xxh3_64(str(item), seed=seed).intdigest() % g
            hash_other_projection_list[hash_result] += 1
        score = hash_projection_list - np.abs(num_map_AO - hash_projection_list - hash_other_projection_list)
        current_best_score = np.max(score)
        max_indices = np.where(score == current_best_score)[0]
        current_max_target_mapped = hash_projection_list[max_indices]
        current_untarget_mapped = hash_other_projection_list[max_indices]
        current_hash_value = max_indices
        current_gap = np.abs(num_map_AO - current_max_target_mapped - current_untarget_mapped)
        if current_best_score > best_score:
            best_seed = seed
            best_score = current_best_score
            best_hash_value = current_hash_value
            best_gap = current_gap
            best_target_mapped = current_max_target_mapped
    if best_seed == -1:
        return -1, 0.0, None
    return best_seed, best_gap, best_target_mapped, best_hash_value


def process_attacker_User(i, n, ratio, target_set, g, domain, splits, e, h_ao):
    '''
    Function to process each attacker in parallel.
    Each attacker finds their optimal hash function.
    :param i: Index of the attacker.
    :param n: Total number of users.
    :param ratio: Ratio of attackers to total users.
    :param target_set: Set of target items.
    :param g: Range of hash function outputs (modulo value).
    :param domain: Total domain size.
    :param splits: Number of target items each attacker considers.
    '''
    average_project_hash = int(domain / g)
    vector = np.zeros(domain, dtype=int)
    if splits < average_project_hash:
        # Split the target set for each user
        splits_list = random.sample(list(target_set), splits)
        # Gap between average mapping
        num_map = average_project_hash
        # Remaining set (unused in this snippet but kept for completeness)
        remaining_set = set(range(domain)) - set(target_set)
        # Adaptive gap between average mapping
        num_map_AO = random.randint(num_map - h_ao, num_map + h_ao)
        # theoretical APA use the omega_list to replace the num_map_AO num_map_AO = np.random.choice([i for i in range(domain)], construct_omega(e, domain, 'OLH_User'))
        #num_map_AO = np.random.choice([i for i in range(domain)], construct_omega(e, domain, 'OLH_User'))
        seed_list = random.sample(range(1, 10000000), num_samples)
        best_seed, best_gap, current_max_target_mapped, best_hash_value = find_hash_function(seed_list, splits_list,
                                                                                             remaining_set, g,
                                                                                             num_map_AO)
    else:
        print('splits > averge_project_hash')
        exit(0)
    # Calculate the index in User_Seed to update
    index = int(n * (1 - ratio) + i)
    for v in range(domain):
        hashed_value = xxhash.xxh3_64(str(v), seed=int(best_seed)).intdigest() % g
        if hashed_value == best_hash_value:
            vector[v] = 1
    print(f'attacker:{i}, target_map:{current_max_target_mapped}, diff:{best_gap}, h_ao:{h_ao}, splits:{splits}')
    return index, vector


def process_attacker_server(i, n, ratio, target_set, g, domain, User_Seed, splits):
    '''
    Function to process each attacker.
    Each attacker finds the best hash value and constructs a vector based on it.

    :param i: Index of the attacker.
    :param n: Total number of users.
    :param ratio: Ratio of attackers to total users.
    :param target_set: Set of target items.
    :param g: Range of hash function outputs (modulo value).
    :param domain: Total domain size.
    :param User_Seed: List of hash seeds for users.
    :return: Tuple of (index in User_Seed, attack_vector).
    '''
    # Calculate the index in User_Seed
    index = int(n * (1 - ratio) + i)
    user_seed = User_Seed[index]

    # Compute hash values for all target items
    target_hashes = {}
    splits_list = random.sample(list(target_set), splits)
    for t in splits_list:
        hashed_value = xxhash.xxh3_64(str(t), seed=int(user_seed)).intdigest() % g
        if hashed_value in target_hashes:
            target_hashes[hashed_value] += 1
        else:
            target_hashes[hashed_value] = 1

    # Find the hash value that maps the most target items
    best_hashed_value = max(target_hashes, key=target_hashes.get)
    max_target_count = target_hashes[best_hashed_value]

    # Construct the attack vector
    attack_vector = np.zeros(domain)
    for v in range(domain):
        hashed_value = xxhash.xxh3_64(str(v), seed=int(user_seed)).intdigest() % g
        if hashed_value == best_hashed_value:
            attack_vector[v] = 1

    print(f'attacker:{i}, best_hashed_value:{best_hashed_value}, max_targets_mapped:{max_target_count}')
    return index, attack_vector


def process_user_seeds(i, User_Seed_noattack, Y_Nattack, domain, g):
    # print(i)
    local_estimate = np.zeros(domain)
    user_seed = User_Seed_noattack[i]
    for v in range(domain):
        if Y_Nattack[i] == (xxhash.xxh3_64(str(v), seed=int(user_seed)).intdigest() % g):
            local_estimate[v] += 1
    # Apply the correction factor
    local_estimate = local_estimate
    return local_estimate


def build_support_list_1_OLH(domain, Y, n, User_Seed, ratio, g, target_set, p, splits, e, h_ao=0, processor=100):
    '''
    build the support list matrix
    :return:
    '''
    # Because the total number of users n is very large, performing a serial search
    # over all hash functions would be extremely time‑consuming. Instead, we use
    # numerical simulation to select the optimal hash. In a real‑world deployment,
    # it remains practical for each user to sample up to 1,000,000 hash functions
    # when searching for the best one.
    K_values, K_probs = calculate_prob_according_sample_size(num_samples, domain, g, h_ao, target_set, splits)

    # Prepare the partial function with fixed arguments for multiprocessing
    process_attacker_partial = partial(

        # Real 1,000,000 hash samples
        #process_attacker_User,
        # Simulate 1,000,000 hash samples, fast version
        process_attacker,
        n=n,
        ratio=ratio,
        target_set=target_set,
        g=g,
        domain=domain,
        splits=splits,
        h_ao= 10*h_ao,
        e=e,
        K_values = K_values,
        K_probs = K_probs
    )

    # Calculate the number of attackers
    num_attackers = int(round(n * ratio))

    # Parallel execution of process_attacker using multiprocessing
    with Pool(processes=processor) as pool:
        # Use imap to process in parallel and tqdm for progress bar
        results = list(tqdm(
            pool.imap(process_attacker_partial, range(num_attackers)),
            total=num_attackers,
            desc='Finding optimal seeds'
        ))

    vector_matrix = np.zeros((num_attackers, domain))
    # Update User_Seed with the results from all attackers
    for i, (index, best_vector) in enumerate(results):
        vector_matrix[i, :] = best_vector

    # Create a partial function with fixed arguments for processing user seeds
    process_partial = partial(
        process_user_seeds,
        User_Seed_noattack=User_Seed,
        Y_Nattack=Y,
        domain=domain,
        g=g
    )

    # Process user seeds across multiple processes
    with Pool(processes=processor) as pool:
        estimates = pool.map(process_partial, range(n - num_attackers))
    # input attack's estimate
    '''with Pool(processes=processor) as pool:
        estimates_input = pool.map(process_partial, range(int(n)))'''

    # Combine the results from all processes
    estimates = np.array(estimates)
    # estimates_input = np.array(estimates_input)
    # estimates_input = estimates_input.reshape(int(n), domain)
    estimates = np.vstack((estimates, vector_matrix))
    estimates = estimates.reshape(int(n), domain)
    Results_support = estimates
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = np.sum(Results_support, axis=0)
    # Estimations_input = np.sum(estimates_input, axis=0)
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    Estimations = a * Estimations - b
    # Estimations_input = a * Estimations_input - b
    Estimations_input = None
    return Results_support, Results_support_one_list, Estimations, Estimations_input


def build_support_list_1_OLH_Server(domain, Y, n, User_Seed, ratio, g, target_set, p, splits, h_ao=0, processor=100):
    '''
    build the support list matrix
    :return:
    '''
    # Prepare the partial function with fixed arguments for multiprocessing
    process_attacker_partial = partial(
        process_attacker_server,
        n=n,
        ratio=ratio,
        target_set=target_set,
        g=g,
        domain=domain,
        User_Seed=User_Seed,
        splits=splits
    )

    # Calculate the number of attackers
    num_attackers = int(round(n * ratio))
    num_normal = int(n - num_attackers)

    # Parallel execution of process_attacker using multiprocessing
    with Pool(processes=processor) as pool:
        # Use imap to process in parallel and tqdm for progress bar
        results = list(tqdm(
            pool.imap(process_attacker_partial, range(num_attackers)),
            total=num_attackers,
            desc='Processing attackers'
        ))

    vector_matrix = np.zeros((num_attackers, domain))
    # Update User_Seed with the results from all attackers
    for i, (index, best_vector) in enumerate(results):
        vector_matrix[i, :] = best_vector

    # Create a partial function with fixed arguments for processing user seeds
    process_partial = partial(
        process_user_seeds,
        User_Seed_noattack=User_Seed,
        Y_Nattack=Y,
        domain=domain,
        g=g
    )

    # Process user seeds across multiple processes
    with Pool(processes=processor) as pool:
        estimates = pool.map(process_partial, range(int(num_normal)))
    # input attack's estimate
    '''with Pool(processes=processor) as pool:
        estimates_input = pool.map(process_partial, range(int(n)))'''

    # Combine the results from all processes
    estimates = np.array(estimates)
    # estimates_input = np.array(estimates_input)
    # estimates_input = estimates_input.reshape(int(n), domain)
    estimates = np.vstack((estimates, vector_matrix))
    estimates = estimates.reshape(int(n), domain)
    Results_support = estimates
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = np.sum(Results_support, axis=0)
    # Estimations_input = np.sum(estimates_input, axis=0)
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    Estimations = a * Estimations - b
    # Estimations_input = a * Estimations_input - b
    Estimations_input = None
    return Results_support, Results_support_one_list, Estimations, Estimations_input


def uniform_sampling_best_vector(target_set, g, d, m, num_samples):
    '''
    Generate multiple binary vectors and find the best one based on two conditions:
    1. The number of 1's in the vector is as close to m as possible (this is the primary condition).
    2. The vector maps as many items from target_set to 1 as possible.

    :param target_set: Set of target item positions (indices).
    :param g: Range of hash function outputs (modulo value, not used here).
    :param d: Desired length of the output vector.
    :param m: Desired number of 1's in the vector.
    :param num_samples: Number of binary vectors to generate.
    :return: The best binary vector of length d.
    '''
    best_vector = None
    closest_ones_diff = float('inf')
    max_target_count = 0
    best_score = -float('inf')
    current_target = None
    current_diff = None

    for _ in range(num_samples):
        # Generate uniform samples for binary vectors (0 or 1)
        vector = np.random.binomial(1, 1 / g, size=d)

        # Count the number of 1's in the vector
        ones_count = np.sum(vector)
        ones_diff = abs(ones_count - m)  # Difference between current 1's count and target m

        # Count how many target items map to positions with 1's in the vector
        target_count = sum(1 for item in target_set if vector[item % d] == 1)

        # Calculate the score: target_count - ones_diff
        score = target_count - ones_diff

        # Update the best vector if the score is better
        if score > best_score:
            best_score = score
            best_vector = vector
            current_target = target_count
            current_diff = ones_diff

    return best_vector, current_target, current_diff


def build_support_list_1_OUE(estimates, n, epsilon):
    '''
    build the support list for OUE
    :param estimates:
    :return:
    '''
    q_OUE = 1 / (math.exp(epsilon) + 1)
    p = 0.5
    q = q_OUE
    Results_support = np.array(estimates)
    Estimations = np.sum(Results_support, axis=0)
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = [(i - n * q_OUE) / (p - q_OUE) for i in Estimations]
    Estimations_input = None

    return Results_support, Results_support_one_list, Estimations, Estimations_input


import scipy.stats as stats


def calculate_prob_according_sample_size(num_samples, d, g, h, target_set, splits):

    splits_list = random.sample(list(target_set), splits)
    target_set = splits_list
    user_vectors = []
    p = 1 / g

    mu = d * p
    sigma = np.sqrt(d * p * (1 - p))

    lower_bound = max(0, mu - h)
    upper_bound = min(d, mu + h)

    binom_dist = stats.binom(d, p)
    ratio = (binom_dist.cdf(upper_bound) - binom_dist.cdf(lower_bound - 1))
    ratio = ratio / (2 * h + 1)
    # ratio = 1

    N_effective = num_samples * ratio
    print('N_effective: ', N_effective)

    K_min = 1
    K_max = len(target_set)
    for K in range(K_max, K_min - 1, -1):
        prob = (p) ** K * N_effective
        if prob < 1:
            K_max = K
        else:
            break
    K_min = max(K_max, 1)

    K_values = np.arange(K_min, len(target_set) + 1)
    K_probs = []
    for K in K_values:
        prob = (p) ** K * N_effective
        K_probs.append(prob)
    K_probs = np.array(K_probs)

    K_probs = K_probs / np.sum(K_probs)

    return K_values, K_probs









