import time

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from scipy.special import betainc
import math
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori
from joblib import Parallel, delayed

chunk_size = 20000
def process_chunk(chunk, min_support=0.04):
    chunk = chunk.astype(bool)
    # Run Apriori on each chunk
    return fpgrowth(chunk, min_support=min_support, use_colnames=True, max_len=12)

def process_chunk_HST(chunk, min_support=0.04):
    chunk = chunk.astype(bool)
    # Run Apriori on each chunk
    return fpgrowth(chunk, min_support=min_support, use_colnames=True, max_len=10)

def process_itemset_chunk_OUE(chunk, n, p, q, eta, support_matrix):
    abnormal_itemsets = []
    users_set = set()
    print('length: ', len(chunk))
    for _, row in tqdm(chunk.iterrows(), desc = 'filtrate the itemsets'):
        z = len(row['itemsets'])
        tau_z_min = n * p * (q ** (z - 1))

        for tau_z in range(int(tau_z_min) + 1, n + 1):
            fpr_bound = (n * p * q ** (z - 1) * (1 - p * q ** (z - 1))) / ((tau_z - n * p * q ** (z - 1)) ** 2)

            if fpr_bound <= eta:
                # Find users who support this set
                itemset = row['itemsets']
                mask = support_matrix[itemset].all(axis=1)
                users = support_matrix[mask].index.tolist()

                if row['support'] >= tau_z / n:
                    abnormal_itemsets.append(row)
                    users_set.update(users)
                break  # Stop the loop after finding a tau_z that satisfies the condition

    return abnormal_itemsets, list(users_set)

def detect_abnormal_itemsets_OUE_parallel(itemsets, n, p, q, eta, support_matrix, n_jobs=1):
    chunks = np.array_split(itemsets, n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(process_itemset_chunk_OUE)(chunk, n, p, q, eta, support_matrix) for chunk in chunks)

    non_empty_results = [pd.DataFrame(r[0]) for r in results if r[0]]
    if non_empty_results:
        abnormal_itemsets = pd.concat(non_empty_results)
    else:
        abnormal_itemsets = pd.DataFrame()

    abnormal_users = set()
    for r in results:
        abnormal_users.update(r[1])

    return abnormal_itemsets, list(abnormal_users)



def process_itemset_chunk(chunk, n, q, eta, support_matrix):
    abnormal_itemsets = []
    users_set = set()

    def calculate_tau_z(z, n, q, eta):
        for tau_z in range(1, n + 1):
            I_value = betainc(tau_z, n - tau_z + 1, q ** (z - 1))
            if I_value <= eta:
                return tau_z
        return None
    print('length: ', len(chunk))
    for _, row in tqdm(chunk.iterrows(), desc = 'filtrate the itemsets'):
        z = len(row['itemsets'])
        tau_z = calculate_tau_z(z, n, q, eta)

        if tau_z is not None:
            # Find users who support this set
            itemset = row['itemsets']
            mask = support_matrix[itemset].all(axis=1)
            users = support_matrix[mask].index.tolist()

            if row['support'] >= tau_z / n:
                abnormal_itemsets.append(row)
                users_set.update(users)

    return abnormal_itemsets, list(users_set)

def detect_abnormal_itemsets_OLH_parallel(itemsets, n, q, eta, support_matrix, n_jobs=1):
    chunks = np.array_split(itemsets, n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(process_itemset_chunk)(chunk, n, q, eta, support_matrix) for chunk in chunks)

    non_empty_results = [pd.DataFrame(r[0]) for r in results if r[0]]
    if non_empty_results:
        abnormal_itemsets = pd.concat(non_empty_results)
    else:
        abnormal_itemsets = pd.DataFrame()
        
    abnormal_users = set()
    for r in results:
        abnormal_users.update(r[1])

    return abnormal_itemsets, list(abnormal_users)


def freq_itemset_mining_OUE(Support_matrix, n, e, ratio, eta=0.1):
    p = 1 / 2
    q = 1 / (math.exp(e) + 1)
    df = pd.DataFrame(Support_matrix)
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    time1 = time.time()
    results = Parallel(n_jobs=1)(delayed(process_chunk)(chunk, q) for chunk in chunks)
    frequent_itemsets = pd.concat(results)
    filtered_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: 4 <= len(x) <= 12)]
    time2 = time.time()
    print('fpgrowth times: ', time2 - time1)
    abnormal_itemsets_OUE, abnormal_users_OUE = detect_abnormal_itemsets_OUE_parallel(filtered_itemsets, n, p, q, eta, df)
    abnormal_users_OUE = np.array(abnormal_users_OUE)
    if ratio != 0:
        FNR = (n * ratio - len(abnormal_users_OUE[abnormal_users_OUE >= n * (1 - ratio)])) / (n * ratio)
        FPR = (len(abnormal_users_OUE) - len(abnormal_users_OUE[abnormal_users_OUE >= n * (1 - ratio)])) / (
                    n * (1 - ratio))
        Recall = len(abnormal_users_OUE[abnormal_users_OUE >= n * (1 - ratio)]) / (n * ratio)
        if len(abnormal_users_OUE) == 0:
            Precision = 0
            F1 = 0
        else:
            Precision = len(abnormal_users_OUE[abnormal_users_OUE >= n * (1 - ratio)]) / len(abnormal_users_OUE)
            F1 = 2 * (Precision * Recall) / (Precision + Recall)
    else:
        FNR = 0
        FPR = 0
        Recall = 0
        F1 = 0
        Precision = 0
        if len(abnormal_users_OUE) > 0:
            print('detect failed!')
        else:
            print('Bypass!')
    print("abnormal_itemsets in OUE:")
    print(abnormal_itemsets_OUE)
    print("Detected users in OUE:", abnormal_users_OUE)
    print(f'FPR:{FPR}, FNR:{FNR}')
    print(f'F1:{F1}')
    if len(abnormal_users_OUE) != 0:
        mask = np.ones(len(Support_matrix), dtype=bool)
        mask[abnormal_users_OUE] = False
        support_clean = Support_matrix[mask]
    else:
        support_clean = Support_matrix
    number_of_users = len(support_clean)
    Estimations = np.sum(support_clean, axis=0)
    aggregate_clean = [(i - len(support_clean) * q) / (p - q) for i in Estimations]
    return aggregate_clean, number_of_users, FPR, FNR, F1, Precision, Recall


def freq_itemset_mining_OLH(Support_matrix, n, e, ratio, eta=0.1):
    g = int(round(math.exp(e))) + 1
    p = math.exp(e) / (math.exp(e) + g - 1)
    q = 1 / g
    df = pd.DataFrame(Support_matrix)
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    time1 = time.time()
    results = Parallel(n_jobs=1)(delayed(process_chunk)(chunk, q) for chunk in chunks)
    frequent_itemsets = pd.concat(results)
    filtered_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: 4 <= len(x) <= 12)]
    time2 = time.time()
    print('fpgrowth times: ', time2-time1)
    abnormal_itemsets_OLH, abnormal_users_OLH = detect_abnormal_itemsets_OLH_parallel(filtered_itemsets, n, q, eta, df)
    abnormal_users_OLH = np.array(abnormal_users_OLH)

    if ratio != 0:
        FNR = (n * ratio - len(abnormal_users_OLH[abnormal_users_OLH >= n * (1 - ratio)])) / (n * ratio)
        FPR = (len(abnormal_users_OLH) - len(abnormal_users_OLH[abnormal_users_OLH >= n * (1 - ratio)])) / (n * (1 - ratio))
        Recall = len(abnormal_users_OLH[abnormal_users_OLH >= n * (1 - ratio)]) / (n * ratio)
        if len(abnormal_users_OLH) == 0:
            Precision = 0
            F1 = 0
        else:
            Precision = len(abnormal_users_OLH[abnormal_users_OLH >= n * (1 - ratio)]) / len(abnormal_users_OLH)
            F1 = 2 * (Precision * Recall) / (Precision + Recall)
    else:
        FNR = 0
        FPR = 0
        Recall = 0
        F1 = 0
        Precision = 0
        if len(abnormal_users_OLH) > 0:
            print('detect failed!')
        else:
            print('Bypass!')
    print("abnormal_itemsets in OLH:")
    print(abnormal_itemsets_OLH)
    print("Detected users in OLH:", abnormal_users_OLH)
    print(f'FPR:{FPR}, FNR:{FNR}')
    print(f'F1:{F1}')
    if len(abnormal_users_OLH) != 0:
        mask = np.ones(len(Support_matrix), dtype=bool)
        mask[abnormal_users_OLH] = False
        support_clean = Support_matrix[mask]
    else:
        support_clean = Support_matrix
    number_of_users = len(support_clean)
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * len(support_clean) / (p * g - 1)
    support_clean = support_clean.sum(axis=0)
    aggregate_clean = a * support_clean - b
    return aggregate_clean, number_of_users, FPR, FNR, F1, Precision, Recall


def freq_itemset_mining_HST(Amplify_Support_matrix, n, e, ratio, eta=0.1):
    Original_support_matrix = np.copy(Amplify_Support_matrix)
    Support_matrix = np.zeros_like(Original_support_matrix, dtype=int)
    Support_matrix[Original_support_matrix > 0] = 1
    g = 2
    q = 1 / g
    df = pd.DataFrame(Support_matrix)
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    time1 = time.time()
    results = Parallel(n_jobs=1)(delayed(process_chunk_HST)(chunk, q) for chunk in chunks)
    frequent_itemsets = pd.concat(results)
    filtered_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: 4 <= len(x) <= 12)]
    time2 = time.time()
    print('fpgrowth times: ', time2 - time1)
    abnormal_itemsets_OLH, abnormal_users_OLH = detect_abnormal_itemsets_OLH_parallel(filtered_itemsets, n, q, eta, df)
    abnormal_users_OLH = np.array(abnormal_users_OLH)

    if ratio != 0:
        FNR = (n * ratio - len(abnormal_users_OLH[abnormal_users_OLH >= n * (1 - ratio)])) / (n * ratio)
        FPR = (len(abnormal_users_OLH) - len(abnormal_users_OLH[abnormal_users_OLH >= n * (1 - ratio)])) / (n * (1 - ratio))
        Recall = len(abnormal_users_OLH[abnormal_users_OLH >= n * (1 - ratio)]) / (n * ratio)
        if len(abnormal_users_OLH) == 0:
            Precision = 0
            F1 = 0
        else:
            Precision = len(abnormal_users_OLH[abnormal_users_OLH >= n * (1 - ratio)]) / len(abnormal_users_OLH)
            F1 = 2 * (Precision * Recall) / (Precision + Recall)
    else:
        FNR = 0
        FPR = 0
        Recall = 0
        F1 = 0
        Precision = 0
        if len(abnormal_users_OLH) > 0:
            print('detect failed!')
        else:
            print('Bypass!')
    print("abnormal_itemsets in OLH:")
    print(abnormal_itemsets_OLH)
    print("Detected users in OLH:", abnormal_users_OLH)
    print(f'FPR:{FPR}, FNR:{FNR}')
    print(f'F1:{F1}')
    if len(abnormal_users_OLH) != 0:
        mask = np.ones(len(Support_matrix), dtype=bool)
        mask[abnormal_users_OLH] = False
        support_clean = Original_support_matrix[mask]
    else:
        support_clean = Original_support_matrix
    number_of_users = len(support_clean)
    aggregate_clean = support_clean.sum(axis=0)
    return aggregate_clean, number_of_users, FPR, FNR, F1, Precision, Recall

