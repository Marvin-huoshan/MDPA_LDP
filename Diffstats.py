import numpy as np
import pandas as pd
from scipy.stats import norm, kstest, kurtosis, skew
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
import itertools
from tqdm import tqdm
from scipy.stats import binom, chisquare

def diffstats(e, d, n, one_list, support_list, ratio, file_name, target_set_size, fix_ratio, ESTIMATE_DIST_Nattack, perturb_method):
    '''
    base on the one numbers for each user to detect the anomalies
    :return:
    '''
    p_binomial = None
    if perturb_method == 'OUE':
        p = 1 / 2
        q = 1 / (math.exp(e) + 1)
        expected_ones = p + (d - 1) * q
        p_binomial = (1 / d) * (p + (d - 1) * q)
    elif perturb_method in ('OLH_User', 'OLH_Server'):
        g = int(round(math.exp(e))) + 1
        p = math.exp(e) / (math.exp(e) + g - 1)
        q = 1 / g
        expected_ones = p + (d - 1) * q
        p_binomial = (1 / d) * (p + (d - 1) * q)
    elif perturb_method in ('HST_Server', 'HST_User'):
        p_binomial = 1 / 2
        p = q = 1 / 2
        expected_ones = d * p_binomial

    data = one_list
    # transfer to DataFrame
    df = pd.DataFrame(data, columns=['Number_of_Ones'])
    # generate support list
    if perturb_method == 'HST_Server' or 'HST_User':
        support = np.zeros_like(support_list, dtype=int)
        support[support_list > 0] = 1
    else:
        support = support_list

    values, counts = np.unique(df['Number_of_Ones'], return_counts=True)
    values = values.astype(int)
    observed_freq = counts / len(data)
    # only focus on exist k
    k_values = values

    # theoretical Binomial
    theoretical_pdf = binom.pmf(k_values, d, p_binomial)
    theoretical_freq = theoretical_pdf * d

    # Discrepancies
    difference = np.abs(observed_freq - theoretical_freq / len(data))
    diff_sorted = sorted(difference, reverse=True)
    #top_n_diff = diff_sorted[:int(std_theoretical)]
    top_n_diff = diff_sorted[:10]
    top_n_diff.reverse()
    start = top_n_diff[0]

    thresholds = np.linspace(start, 1, 100)
    best_threshold = None
    best_combination = None
    best_ks_stat = float('inf')
    results = []
    all_anomalies = set()
    to_remove = []
    best_metric = float('inf')
    for threshold in top_n_diff:

        df_temp = df.copy()

        # observation frequency
        values, counts = np.unique(df_temp['Number_of_Ones'], return_counts=True)
        values = values.astype(int)
        observed_freq = counts / len(data)
        observed_count = observed_freq * len(data)
        '''observed_freq = np.zeros(len(k_values))
        observed_freq[values] = counts
        observed_freq /= len(data)'''

        # Binomial
        theoretical_pdf = binom.pmf(k_values, d, p_binomial)
        theoretical_freq = theoretical_pdf * d

        # Discrepancies
        difference = np.abs(observed_freq - theoretical_freq / len(data))
        difference_dict = dict(zip(values, difference))


        # Flagging and removing data points that cause large discrepancies
        df_temp['Is_Anomaly'] = df_temp['Number_of_Ones'].isin(values[difference > threshold])
        anomalies_index = df_temp.index[df_temp['Is_Anomaly']].tolist()
        support_sum = np.sum(support[anomalies_index], axis=0)

        top_n = 6
        if len(support_sum) <= top_n:
            valid_support_indices = list(np.argsort(support_sum)[::-1])
        else:
            valid_support_indices = list(np.argsort(support_sum)[-top_n:][::-1])
        all_combinations = []
        for r in range(1, len(valid_support_indices) + 1):
            all_combinations.extend(list(itertools.combinations(valid_support_indices, r)))
        print('combinations: ', all_combinations)
        print('values: ', values[difference > threshold])
        temp_anomalies = set()
        if len(all_combinations) == 0:
            temp_anomalies = set()
            #mean = np.mean(df_temp['Number_of_Ones'])
            #std = np.std(df_temp['Number_of_Ones'])
            #ks_stat, p_value = kstest(df_temp['Number_of_Ones'], 'norm', args=(mean, std))
            expected_counts = [len(df_temp) * binom.pmf(k, d, p_binomial) for k in k_values]
            chi2_stat, p_value = chisquare(observed_count, f_exp=expected_counts)
            kurtosiss = kurtosis(df_temp['Number_of_Ones'])
            skewness = skew(df_temp['Number_of_Ones'])
            ks_stat = chi2_stat
            #combine_metric = ks_stat + 1.1 * abs(kurtosiss) + skewness
            combine_metric = ks_stat + p_value
            if combine_metric < best_metric:
                best_metric = combine_metric
                best_threshold = threshold
                best_ks = ks_stat
                best_p = p_value
                best_kur = kurtosiss
                best_ske = skewness
                best_combination = []
                best_mal = []
                print('best thre: ', best_threshold)
                print('best combine: ', best_combination)
            continue

        for combination in tqdm(all_combinations):
            df_temp_in = df_temp.copy()
            temp_anomalies = set()
            to_remove = [i for i in anomalies_index if any(support[i][idx] == 1 for idx in combination)]
            to_remove_array = np.array(to_remove)
            if ratio != 0:
                malicious_ratio = (len(to_remove_array[to_remove_array >= n * (1 - ratio)])) / (n * ratio)
            else:
                malicious_ratio = -1
            temp_anomalies.update(to_remove)
            df_temp_comb = df_temp_in.drop(to_remove)
            cleaned_data = df_temp_comb['Number_of_Ones']
            values_c, counts_c = np.unique(cleaned_data, return_counts=True)
            values_c = values_c.astype(int)
            observed_count = counts_c
            index_1 = np.where(difference > threshold)
            amends = amend(d, e, p, q, len(cleaned_data), len(index_1))
            for ix in index_1:
                observed_count[ix] += int(amends)
            expected_counts = [len(cleaned_data) * binom.pmf(k, d, p_binomial) for k in k_values]
            observed_total = sum(observed_count)
            expected_total = sum(expected_counts)
            adjusted_expected_counts = [count * observed_total / expected_total for count in expected_counts]
            test = [(observed_count[i] - adjusted_expected_counts[i])**2 / adjusted_expected_counts[i] for i in range(len(observed_count))]
            chi2_stat_cleaned, p_value_cleaned = chisquare(observed_count, f_exp=adjusted_expected_counts)
            kurtosis_cleaned = kurtosis(cleaned_data)
            skewness_cleaned = skew(cleaned_data)
            ks_stat_cleaned = chi2_stat_cleaned
            #combine_metric = ks_stat_cleaned + 1.1 * abs(kurtosis_cleaned) + skewness_cleaned
            combine_metric = ks_stat_cleaned + p_value_cleaned

            if combine_metric < best_metric:
                a = to_remove
                best_metric = combine_metric
                best_threshold = threshold
                best_combination = combination
                best_ks = ks_stat_cleaned
                best_p = p_value_cleaned
                best_kur = kurtosis_cleaned
                best_ske = skewness_cleaned
                best_mal = temp_anomalies
                print('best thre: ', best_threshold)
                print('best combine: ', best_combination)
                print(malicious_ratio)
        results.append(
            (threshold, best_ks, best_p, best_kur, best_ske,
                best_mal, best_combination))
    if ratio == 0:
        if len(all_anomalies) == 0:
            print('Bypass')
        else:
            print('Failed')
        
    if len(results) == 0 or best_combination == []:
        print('no anomalies')
        acc3 = ratio
        acc2 = 0
        F1 = 0
        print(f"Number of anomalies: {len(all_anomalies)}")
        print(f'FPR: {acc2}')
        print(f"FNR: {acc3}")
        print(f"F1 of Diffstats: {F1}")
        return ESTIMATE_DIST_Nattack, n, acc2, acc3, F1
    # Output the optimal threshold and the corresponding statistics
    print(f"Best threshold: {best_threshold}")
    print(f"Best combination : {best_combination}")
    for result in results:
        if result[0] == best_threshold and result[6] == best_combination:
            print(
                f"Threshold: {result[0]}, KS stat: {result[1]}, p-value: {result[2]}, Kurtosis: {result[3]}, Skewness: {result[4]}")

    # Merge all known anomalies indexes
    best_anomalies = next((res for res in results if res[0] == best_threshold and res[6] == best_combination), None)[5]
    #best_anomalies = results[[result[6] for result in results].index(best_combination)][5]
    all_anomalies = best_anomalies
    all_anomalies = np.array(list(all_anomalies))

    # all_anomalies = list(all_anomalies)
    if len(all_anomalies) == 0:
        print('no anomalies')
        acc3 = ratio
        acc2 = 0
        Precision = 0
        Recall = 0
        F1 = 0
        print(f"Number of anomalies: {len(all_anomalies)}")
        print(f'FPR: {acc2}')
        print(f"FNR: {acc3}")
        print(f"F1 of Diffstats: {F1}")
    else:
        if ratio != 0:
            acc3 = (n * ratio - len(all_anomalies[all_anomalies >= n * (1 - ratio)])) / (n * ratio)
        else:
            acc3 = 0
        acc2 = (len(all_anomalies) - len(all_anomalies[all_anomalies >= n * (1 - ratio)])) / (n * (1 - ratio))
        Precision = len(all_anomalies[all_anomalies >= n * (1 - ratio)]) / len(all_anomalies)
        Recall = len(all_anomalies[all_anomalies >= n * (1 - ratio)]) / (n * ratio)
        if (Precision + Recall) == 0:
            F1 = 0
        else:
            F1 = 2 * (Precision * Recall) / (Precision + Recall)
        print(f"Number of anomalies: {len(all_anomalies)}")
        print(f"FPR: {acc2}")
        print(f"FNR: {acc3}")
        print(f"F1 of Diffstats: {F1}")
        print(df.loc[all_anomalies])

    # cleaned data
    cleaned_data = df.drop(all_anomalies)['Number_of_Ones']

    mean_cleaned = np.mean(cleaned_data)
    std_cleaned = np.std(cleaned_data)

    # ks cleaned
    ks_stat_cleaned, p_value_cleaned = kstest(cleaned_data, 'norm', args=(mean_cleaned, std_cleaned))
    print(f"KS test statistic after cleaning: {ks_stat_cleaned}, p-value: {p_value_cleaned}")

    kurtosis_cleaned = kurtosis(cleaned_data)
    skewness_cleaned = skew(cleaned_data)
    #print(f"Kurtosis after cleaning: {kurtosis_cleaned}, Skewness after cleaning: {skewness_cleaned}")

    plt.figure(figsize=(12, 6))
    sns.histplot(df['Number_of_Ones'], kde=True, bins=30)
    plt.title('Distribution of Number of Ones Reported by Users (Not Cleaned)')
    plt.axvline(x=expected_ones, color='red', linestyle='--', label=f'Theoretical Mean: {expected_ones}')
    plt.legend()

    plt.figure(figsize=(12, 6))
    sns.histplot(cleaned_data, kde=True, bins=30)
    plt.title('Distribution of Number of Ones Reported by Users (Cleaned)')
    plt.axvline(x=expected_ones, color='red', linestyle='--', label=f'Theoretical Mean: {expected_ones}')
    plt.legend()
    #plt.show()

    thresholds, ks_stats, p_values, kurtoses, skewnesses, _, _ = zip(*results)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(thresholds, ks_stats, label='KS Statistic')
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('KS Statistic')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(thresholds, p_values, label='p-value')
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('p-value')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(thresholds, kurtoses, label='Kurtosis')
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Kurtosis')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(thresholds, skewnesses, label='Skewness')
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Skewness')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    if len(all_anomalies) != 0:
        mask = np.ones(len(support), dtype=bool)
        mask[all_anomalies] = False
        support_clean = support_list[mask]
    else:
        support_clean = support_list
    number_of_users = len(support_clean)
    support_clean = support_clean.sum(axis=0)
    #print('support_clean: ', support_clean)
    if perturb_method == 'OUE':
        aggregate_clean = (support_clean - number_of_users * q) / (p - q)
        #print('aggregate_clean: ', aggregate_clean)
    elif perturb_method in ('OLH_User', 'OLH_Server'):
        a = 1.0 * g / (p * g - 1)
        b = 1.0 * len(support_clean) / (p * g - 1)
        aggregate_clean = a * support_clean - b
    elif perturb_method in ('HST_Server', 'HST_User'):
        aggregate_clean = support_clean
    return aggregate_clean, number_of_users, acc2, acc3, F1, Precision, Recall


def amend(d, e, p, q, n, k_size):

    p_prime = (p + (d - 1) * q) / d

    # Calculate probability of exactly 27 successes
    #prob_k = binom.pmf(int(d * p_prime), d, p_prime)
    lower_bound = int(d * p_prime - math.floor(k_size / 2))
    upper_bound = int(d * p_prime + math.floor(k_size / 2))

    # Calculate the interval probabilities, using the CDF
    prob_interval = binom.cdf(upper_bound, d, p_prime) - binom.cdf(lower_bound - 1, d, p_prime)

    expected_occurrences = n * prob_interval * q

    return expected_occurrences / k_size
