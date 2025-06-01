import numpy as np
import random
import math
from Poisoning.APA import construct_omega

def HST_Users(X, ratio, domain, epsilon, n, target_set, h_ao, splits):
    '''
    Perform the HST protocol
    :param X: The real values for each users
    :param ratio: fake users ratio
    :param domain: domain size
    :param epsilon: privacy budget
    :param n: number of users
    :param target_set: fake users target set
    :param h_ao: APA's parameter
    :param splits: MGA-A's parameter
    :return: support_list, one_list, ESTIMATE_DIST, ESTIMATE_Input
    '''
    c = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
    s_vectors = np.zeros((n, domain))
    average_1_num = domain / 2
    fake_user_num = int(round(n * ratio))
    normal_user_num = n - fake_user_num
    start_idx = n - fake_user_num
    h_ao *= 10
    y_values = np.zeros(n)
    # theoretical APA use the omega_list to replace the averge_1_num and set h_ao = 0
    '''if h_ao != 0:
        average_1_num = np.random.choice([i for i in range(domain)], construct_omega(epsilon, domain, 'OUE'))
        h_ao = 0'''
    for i in range(normal_user_num):
        # User's true data item
        v = X[i]  # Assuming X[i] is in the range [0, domain-1]
        # Generate random public vector s_j
        s_i = np.random.choice([-1.0, 1.0], size=domain)
        s_vectors[i, :] = s_i
        # Get s_j[v_b]
        s_i_v = s_i[v]
        # Perturbation process
        if random.random() < math.exp(epsilon) / (math.exp(epsilon) + 1):
            y = c * s_i_v
        else:
            y = -c * s_i_v
        y_values[i] = y
    for i in range(fake_user_num):

        splits_list = random.sample(list(target_set), splits)
        local_user_data = np.full(domain, -1.0)
        local_user_data[list(splits_list)] = 1
        remaining_set = list(set(range(domain)) - set(splits_list))
        diff = int(average_1_num - len(splits_list))
        diff_AO = random.randint(diff - h_ao, diff + h_ao)
        #print(f'attacker:{i}, exp1:{average_1_num}, h_ao:{h_ao}, splits:{splits}')
        if diff_AO > 0 and len(remaining_set) >= diff:
            random_numbers = random.sample(remaining_set, diff_AO)
            local_user_data[random_numbers] = 1
        idx = start_idx + i
        s_vectors[idx,:] = local_user_data
        y = c
        y_values[idx] = y

    support_list = y_values.reshape(-1, 1) * s_vectors
    ESTIMATE_DIST = np.sum(support_list, axis=0)
    Results_support_one_list = np.sum(s_vectors == 1, axis=1)

    return support_list, Results_support_one_list, ESTIMATE_DIST, ESTIMATE_DIST


def HST_Server(X, ratio, domain, epsilon, n, target_set, splits):
    '''
    Perform the HST protocol
    :param X: The real values for each users
    :param ratio: fake users ratio
    :param domain: domain size
    :param epsilon: privacy budget
    :param n: number of users
    :param target_set: fake users target set
    :return: support_list, one_list, ESTIMATE_DIST, ESTIMATE_Input
    '''
    c = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
    s_vectors = np.zeros((n, domain))
    fake_user_num = int(round(n * ratio))
    normal_user_num = n - fake_user_num
    start_idx = n - fake_user_num
    y_values = np.zeros(n)
    for i in range(n):
        # Generate random public vector s_i
        s_i = np.random.choice([-1.0, 1.0], size=domain)
        s_vectors[i, :] = s_i
    for i in range(normal_user_num):
        # User's true data item
        v = X[i]  # Assuming X[i] is in the range [0, domain-1]
        # Generate random public vector s_j
        s_i = s_vectors[i, :]
        # Get s_j[v_b]
        s_i_v = s_i[v]
        # Perturbation process
        if random.random() < math.exp(epsilon) / (math.exp(epsilon) + 1):
            y = c * s_i_v
        else:
            y = -c * s_i_v
        y_values[i] = y
    splits_list = random.sample(list(target_set), splits)
    for i in range(fake_user_num):
        idx = start_idx + i
        s_i = s_vectors[idx, :]
        positive_count = 0
        negative_count = 0
        for v in splits_list:
            if s_i[v] == 1.0:
                positive_count += 1
            elif s_i[v] == -1.0:
                negative_count += 1
        if positive_count >= negative_count:
            y = c
        else:
            y = -c
        y_values[idx] = y
        print(f'Attacker {i}, idx: {idx}, positive_count: {positive_count}, negative_count: {negative_count}, y: {y}')

    support_list = y_values.reshape(-1, 1) * s_vectors
    ESTIMATE_DIST = np.sum(support_list, axis=0)
    Results_support_one_list = np.sum(s_vectors == 1, axis=1)

    return support_list, Results_support_one_list, ESTIMATE_DIST, ESTIMATE_DIST