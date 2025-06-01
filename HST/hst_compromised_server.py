import numpy as np
from scipy.stats import wasserstein_distance
import csv
import sys
sys.path.append("../attack_mean/")
from dist_shift_measure import shift_measure
import copy


def calc_mean(dist, bin, size):
    bin_width = 1 / bin
    sampled_data = np.array([])
    for i in range(len(dist)):
        if dist[i] >= 0:
            sampled_data = np.append(sampled_data, np.random.uniform(i * bin_width, (i + 1) * bin_width, int(size * dist[i])))
        else:
            sampled_data = np.append(sampled_data, -1 * np.random.uniform(i * bin_width, (i + 1) * bin_width, int(size * -dist[i])))
    return np.mean(sampled_data)

def RR(eps, value):
    randoms = np.random.uniform(0, 1)
    if randoms < np.exp(eps) / (np.exp(eps) + 1):
        return (np.exp(eps) + 1) / (np.exp(eps) - 1) * value
    else:
        return -(np.exp(eps) + 1) / (np.exp(eps) - 1) * value


def HST(ori_samples, l, h, eps, target_value, fake_user_num, post_type, domain_bins):
    samples = (ori_samples - l) / (h - l)
    np.random.shuffle(samples)
    n = np.size(samples)
    bin_width = 1 / domain_bins
    index = samples / bin_width
    index = index.astype("int")
    mark = index == domain_bins
    index[mark] -= 1

    noisy_result = np.random.choice([-1., 1.], [n, domain_bins], replace=True)
    # print(noisy_result[-fake_user_num:, :])


    if np.size(target_value) != 0:
        noisy_bit = np.array([noisy_result[i, index[i]] for i in range(n)])

        # perturb
        randoms = np.random.uniform(0, 1, len(noisy_bit))
        mark = randoms < np.exp(eps) / (np.exp(eps) + 1)

        noisy_bit[mark] = (np.exp(eps) + 1) / (np.exp(eps) - 1) * noisy_bit[mark]

        mark = randoms > np.exp(eps) / (np.exp(eps) + 1)
        noisy_bit[mark] = -(np.exp(eps) + 1) / (np.exp(eps) - 1) * noisy_bit[mark]

        compromised_user_noisy_result = noisy_result[-fake_user_num:, :]
        indicator = []
        for x in compromised_user_noisy_result:
            idx_one_mean = np.mean(np.where(x == 1)[0])
            idx_minus_one_mean = np.mean(np.where(x == -1)[0])
            # print(idx_one_mean)
            # print(idx_minus_one_mean)
            if idx_one_mean >= idx_minus_one_mean:
                indicator.append(1)
            else:
                indicator.append(-1)
        indicator = np.array(indicator).reshape([fake_user_num, 1])

        noisy_bit = noisy_bit.reshape([n, 1])

        noisy_bit[-fake_user_num:] = noisy_result[-fake_user_num:, -1].reshape([fake_user_num, 1]) * (np.exp(eps) + 1) / (np.exp(eps) - 1)
        # noisy_bit[-fake_user_num:] = (np.exp(eps) + 1) / (np.exp(eps) - 1) * indicator

        # print(noisy_bit[-fake_user_num:])

        # print(noisy_result[-fake_user_num:, -1].reshape([fake_user_num, 1]))
        # print(noisy_bit[-fake_user_num:])

    else:
        noisy_bit = np.array([noisy_result[i, index[i]] for i in range(n)])

        # perturb
        randoms = np.random.uniform(0, 1, len(noisy_bit))
        mark = randoms < np.exp(eps) / (np.exp(eps) + 1)

        noisy_bit[mark] = (np.exp(eps) + 1) / (np.exp(eps) - 1) * noisy_bit[mark]

        mark = randoms > np.exp(eps) / (np.exp(eps) + 1)
        noisy_bit[mark] = -(np.exp(eps) + 1) / (np.exp(eps) - 1) * noisy_bit[mark]
        noisy_bit = noisy_bit.reshape([n, 1])

    # aggregation
    result = noisy_result * noisy_bit
    result = np.sum(result, axis=0)

    if post_type == 0:
        return result
    if post_type == 1:
        return norm_sub(n, result)
    if post_type == 2:
        return norm(n, result)
    if post_type == 3:
        return norm_mul(n, result)
    if post_type == 4:
        return norm_cut(n, result)

def norm_sub(n, est_dist):  # this is count, not frequency
    print("norm_sub")
    while (np.fabs(sum(est_dist) - n) > 1) or (est_dist < 0).any(): # Norm-Sub
        est_dist[est_dist < 0] = 0
        total = sum(est_dist)
        mask = est_dist > 0
        diff = (n - total) / np.sum(mask)
        est_dist[mask] += diff

    return est_dist

def norm(n, est_dist):   # this is count, not frequency
    print("norm")
    estimates = np.copy(est_dist)
    total = sum(estimates)
    domain_size = len(est_dist)
    diff = (n - total) / domain_size
    estimates += diff
    return estimates

def norm_mul(n, est_dist):
    print("norm_mul")
    estimates = np.copy(est_dist)
    estimates[estimates < 0] = 0
    total = sum(estimates)
    return estimates * n / total

def norm_cut(n, est_dist):
    print("norm_cut")
    estimates = np.copy(est_dist)
    order_index = np.argsort(estimates)

    total = 0
    for i in range(len(order_index)):
        total += estimates[order_index[- 1 - i]]
        if total > n:
            break
        if total < n and np.abs(-2-i) == len(order_index):
            return estimates
        if total < n and estimates[order_index[- 2 - i]] <= 0:
            estimates[order_index[:- 1 - i]] = 0
            return estimates * n / total

    for j in range(i + 1, len(order_index)):
        estimates[order_index[- 1 - j]] = 0

    return estimates * n / total


# test HST
if __name__ == "__main__":
    data = np.load("../data/normal_numerical.npy")
    # data = np.load("../data/taxi_numerical.npy")
    # data = np.load("../data/retirement_numerical.npy")
    n = np.size(data)
    l = np.min(data)
    h = np.max(data)
    bins = 32
    eps = 0.1

    data_dist, _ = np.histogram(data, range=[l, h], bins=bins)
    target_value = np.array([1])
    fake_user_num = int(n * 0.05)

    result = HST(data, l, h, eps, target_value, fake_user_num, bins)
    distance = shift_measure.shift_distance(data_dist / n, result / n, bins=512)
    print("distance after attack:", distance)


    fake_data = copy.copy(data)
    fake_data[-fake_user_num:] = h
    fake_data_dist, _ = np.histogram(fake_data, range=[l, h], bins=bins)
    base_line_distance = shift_measure.shift_distance(data_dist / n, fake_data_dist / n, bins=512)
    print("baseline attack distance:", base_line_distance)

    import matplotlib.pyplot as plt
    bin_index = np.linspace(0, 1, bins + 1)[1:] - 1 / (2 * bins)
    plt.bar(bin_index, result / n, width=1 / bins)
    plt.show()