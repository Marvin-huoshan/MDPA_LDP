import argparse
import math

import time

from func_timeout import func_timeout, FunctionTimedOut

from build_support import build_support_list_1_OLH, build_support_list_1_OUE, build_support_list_1_OLH_Server
from HST.HST import HST_Users, HST_Server
import random
from multiprocessing import Pool
import numpy as np
import xxhash

domain = 0
epsilon = 0.0
n = 0
g = 0
X = []
Y = []
Y_Nattack = []
sample = []
REAL_DIST = []
ESTIMATE_DIST = []
User_Seed = []
User_Seed_Nattack = []
p = 0.0
q = 0.0
Gain = 0
splits = 0
h_ao = 0


from tqdm import tqdm


def perturb_ideal(target_set = None, ratio = 0, e = 0):
    '''

    :param ratio: the ratio of fake user
    :return:
    '''
    global Y, Gain, User_Seed
    Y = np.zeros(n)
    #random.shuffle(X)
    for i in range(n):
        if i < n * (1 - ratio):
            v = X[i]
            # intdigest -> get integer value from xxhash(Output of xxhash is a hash item)
            x = (xxhash.xxh3_64(str(v), seed=i).intdigest() % g)
            y = x
            p_sample = np.random.random_sample()
            if p_sample > p - q:
                # perturb
                y = np.random.randint(0, g)
            Y[i] = y
            User_Seed[i] = i
        else:
            projection_total = len(target_set)
            max_seed = i
            v = random.choice(target_set)
            x = (xxhash.xxh3_64(str(v), seed=i).intdigest() % g)
            y = x
            p_sample = np.random.random_sample()
            if p_sample > p - q:
                # perturb
                y = np.random.randint(0, g)
            Y[i] = y
            User_Seed[i] = i
    return User_Seed

def perturb_oue_process(args):
    start, end, n, domain, q_OUE, ratio, target_set, X, h_ao, splits = args
    local_user_data = np.zeros((end - start, domain), dtype=int)
    h_ao *= 10
    averge_1_num = int(0.5 + (domain - 1) * q_OUE)
    for i in tqdm(range(start, end)):
        v = X[i]
        if i < n * (1 - ratio):
            # benign users
            random_flip = np.random.rand(domain) < q_OUE

            local_user_data[i - start, :] = random_flip

            local_user_data[i - start, v] = 1 if np.random.rand() < 0.5 else 0
        else:
            # fake users
            if splits < averge_1_num:
                splits_list = random.sample(list(target_set), splits)
                local_user_data[i - start, list(splits_list)] = 1
                remaining_set = list(set(range(domain)) - set(splits_list))
                diff = int(averge_1_num - len(splits_list))
                diff_AO = random.randint(diff - h_ao, diff + h_ao)
                #print(f'attacker:{i}, exp1:{int(0.5 + (domain - 1) * q_OUE)}, h_ao:{h_ao}, splits:{splits}')
                if diff_AO > 0 and len(remaining_set) >= diff:
                    random_numbers = random.sample(remaining_set, diff_AO)
                    local_user_data[i - start, random_numbers] = 1
            else:
                splits_list = random.sample(list(target_set), averge_1_num)
                local_user_data[i - start, list(splits_list)] = 1
                remaining_set = list(set(range(domain)) - set(splits_list))
                diff = int(averge_1_num - len(splits_list))
                diff_AO = random.randint(diff - h_ao, diff + h_ao)
                #print(f'attacker:{i}, exp1:{int(0.5 + (domain - 1) * q_OUE)}, h_ao:{h_ao}, splits:{splits}')
                if diff_AO > 0 and len(remaining_set) >= diff:
                    random_numbers = random.sample(remaining_set, diff_AO)
                    local_user_data[i - start, random_numbers] = 1
    return local_user_data

def perturb_OUE_multi(target_set = None, ratio = 0, h_ao = 0, split = 0, num_processes=1):
    global p, q
    q_OUE = 1 / (math.exp(epsilon) + 1)
    p = 0.5
    q = q_OUE
    # Split the task into chunks for each process
    ranges = [(i * n // num_processes, (i + 1) * n // num_processes, n, domain, q_OUE, ratio, target_set, X, h_ao, split) for i in range(num_processes)]
    with Pool(num_processes) as pool:
        results = pool.map(perturb_oue_process, ranges)
    # Combine the results from each process
    user_data = np.vstack(results)
    #Estimations = np.sum(user_data, axis=0)
    return user_data


def execute_with_func_timeout(func, timeout_seconds=3600, *args, **kwargs):
    try:
        time1 = time.time()
        _, _, _, _, _, _, _ = func_timeout(timeout_seconds, func, args=args, kwargs=kwargs)
        time2 = time.time()
        return time2 - time1
    except FunctionTimedOut:
        print(f"The function has been executed for more than {timeout_seconds / 3600} hours and has been terminated.")
        return None

def generate_zipf_dist():
    '''
    generate the zipf distribution according the rules.
    :return:
    '''
    global REAL_DIST, domain, sample, X
    sample = np.zeros(n, dtype=np.int32)
    # the parameter of the zipf distribution
    a = 1.5
    top_1000_values = np.arange(1, domain+1)
    probabilities = 1 / top_1000_values ** a
    probabilities /= probabilities.sum()
    samples = np.random.choice(top_1000_values, size=n, p=probabilities)
    samples -= 1
    for i in range(n):
        REAL_DIST[samples[i]] += 1
    X = samples
    '''global X
    X = np.load('./zipf.npy')
    for i in range(n):
        REAL_DIST[X[i]] += 1'''
    #print('REAL: ', REAL_DIST)

def generate_power_dist():
    '''
    generate the zipf distribution according the rules.
    :return:
    '''
    global REAL_DIST, domain, sample, X
    #np.random.seed(0)
    sample = np.zeros(n, dtype=np.int32)
    # the parameter of the zipf distribution
    a = 0.5
    top_1000_values = np.arange(1, domain + 1)
    probabilities = 1 / top_1000_values ** a
    probabilities /= probabilities.sum()
    samples = np.random.choice(top_1000_values, size=n, p=probabilities)
    samples -= 1
    for i in range(n):
        REAL_DIST[samples[i]] += 1
    X = samples

def generate_emoji_dist():

    global X
    data = np.load('datasets/emoji.npy')
    X = np.copy(data)
    for i in range(n):
        REAL_DIST[data[i]] += 1

import pandas as pd

from sklearn.preprocessing import LabelEncoder
def generate_fire_dist():

    global X
    values = pd.read_csv("datasets/fire.csv")["Unit_ID"]
    #print(np.array(values))
    lf = LabelEncoder().fit(values)
    data = lf.transform(values)
    X = np.copy(data)
    for i in range(n):
        REAL_DIST[data[i]] += 1

def generate_auxiliary():
    '''
    domain-> input domain
    :return:
    '''
    global ESTIMATE_DIST, REAL_DIST, n, p, q, epsilon, domain, User_Seed, User_Seed_Nattack, target_set_size, vswhat, g, splits, h_ao
    epsilon = args.epsilon
    splits = args.splits
    h_ao = args.h_ao
    g = int(round(math.exp(args.epsilon))) + 1
    #target_set_size = args.target_set_size
    if args.dataset == 'zipf':
        n = 1000000
        domain = 1024
    if args.dataset == 'emoji':
        n = 218477
        # n = 100000
        domain = 1496
    if args.dataset == 'fire':
        n = 723090
        domain = 296
    if args.dataset == 'kosarak':
        n = 6374415
        domain = 2000
    User_Seed = np.zeros(n)
    User_Seed_Nattack = np.zeros(n)
    REAL_DIST = np.zeros(domain)
    #REAL_DIST = Counter()
    ESTIMATE_DIST = np.zeros(domain)
    #ESTIMATE_DIST = Counter()
    # stay unchanged with probability p; switch to a different value in [g] with probability q;
    p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
    q = 1.0 / (math.exp(epsilon) + g - 1)

    print('n: ', n)
    print('ratio: ', ratio)
    print('target_set_size: ', target_set_size)
    print('e: ', epsilon)
    print('split: ', splits)
    print('h_ao: ', h_ao)
    time.sleep(2)

def main_OAP(e,target_set_size, fix_ratio = 1):
    global Gain, ESTIMATE_DIST, Y
    generate_auxiliary()
    # prepare parameter such as p and q
    # For each user generate X(vector) and counting the number occurrences of each value

    if args.dataset == 'zipf':
        generate_zipf_dist()
    if args.dataset == 'emoji':
        generate_emoji_dist()
    if args.dataset == 'fire':
        generate_fire_dist()
    print(len(X))
    time_diff_1 = 0
    time_diff_2 = 0
    elapsed_time = 0


    for i in range(args.exp_round):
        Gain = 0
        # In each iteration:
        # sample the index of the key of the REAL_DIST
        # get the target_set_size key in REAL_DIST
        target_set = random.sample(range(0, domain), target_set_size)
        # target_set = list(range(domain))[-target_set_size:]
        from FIAD_detect import freq_itemset_mining_OLH, freq_itemset_mining_OUE, freq_itemset_mining_HST
        print('target set: ', target_set)

        print('fix split: ', splits)
        print('fix h_ao: ', h_ao)
        #User_Seed_Nattack = perturb_noattack()
        '''with open('Y' + '_' + str(ratio) + '_' + str(e) + '.pickle', 'rb') as f:
            Y = pickle.load(f)'''
        f_T = sum(REAL_DIST[element] for element in target_set) / sum(REAL_DIST)
        print('Gain: ', Gain)
        Gain = (Gain - n * ratio * (f_T * (p - 1/g) + target_set_size * 1/g)) / (n * (p - 1/g))
        #print('Attack Gain: ', Gain)
        print('protocol: ', protocol)
        time.sleep(2)
        if protocol == 'OLH_User':
            User_Seed = perturb_ideal(target_set, ratio, e)
            perturb_method = 'OLH_User'
            support_list, one_list, ESTIMATE_DIST_attack, ESTIMATE_Input = build_support_list_1_OLH(domain, Y, n, User_Seed, ratio, g, target_set, p, splits, h_ao, processor=2)
            #recover_data_FIAD, recover_n_FIAD, FPR_FIAD_tmp, FNR_FIAD_tmp, F1_FIAD_tmp, Precision_FIAD_tmp, Recall_FIAD_tmp = freq_itemset_mining_OLH(support_list, n, e, ratio)
            elapsed_time = execute_with_func_timeout(freq_itemset_mining_OLH, timeout_seconds=3600, Support_matrix=support_list, n=n, e=e, ratio=ratio)

        elif protocol == 'OLH_Server':
            User_Seed_Server = perturb_ideal(target_set, ratio, e)
            perturb_method = 'OLH_Server'
            support_list, one_list, ESTIMATE_DIST_attack, ESTIMATE_Input = build_support_list_1_OLH_Server(domain, Y, n, User_Seed_Server, ratio, g, target_set, p, splits, h_ao, processor=2)
            #recover_data_FIAD, recover_n_FIAD, FPR_FIAD_tmp, FNR_FIAD_tmp, F1_FIAD_tmp, Precision_FIAD_tmp, Recall_FIAD_tmp = freq_itemset_mining_OLH(support_list, n, e, ratio)
            elapsed_time = execute_with_func_timeout(freq_itemset_mining_OLH, timeout_seconds=3600, Support_matrix=support_list, n=n, e=e, ratio=ratio)


        elif protocol == 'OUE':
            ESTIMATE_DIST_Matrix = perturb_OUE_multi(target_set, ratio, h_ao, splits)
            perturb_method = 'OUE'
            support_list, one_list, ESTIMATE_DIST_attack, ESTIMATE_Input = build_support_list_1_OUE(ESTIMATE_DIST_Matrix, n, epsilon)
            #recover_data_FIAD, recover_n_FIAD, FPR_FIAD_tmp, FNR_FIAD_tmp, F1_FIAD_tmp, Precision_FIAD_tmp, Recall_FIAD_tmp = freq_itemset_mining_OUE(support_list, n, e, ratio)
            elapsed_time = execute_with_func_timeout(freq_itemset_mining_OUE, timeout_seconds=3600, Support_matrix=support_list, n=n, e=e, ratio=ratio)


        elif protocol == 'HST_User':
            perturb_method = 'HST_User'
            support_list, one_list, ESTIMATE_DIST_attack, ESTIMATE_Input = HST_Users(X, ratio, domain, e, n, target_set, h_ao, splits)
            #recover_data_FIAD, recover_n_FIAD, FPR_FIAD_tmp, FNR_FIAD_tmp, F1_FIAD_tmp, Precision_FIAD_tmp, Recall_FIAD_tmp = freq_itemset_mining_HST(support_list, n, e, ratio)
            elapsed_time = execute_with_func_timeout(freq_itemset_mining_HST, timeout_seconds=3600, Amplify_Support_matrix=support_list, n=n, e=e, ratio=ratio)


        elif protocol == 'HST_Server':
            perturb_method = 'HST_Server'
            support_list, one_list, ESTIMATE_DIST_attack, ESTIMATE_Input = HST_Server(X, ratio, domain, e, n, target_set, splits)
            #recover_data_FIAD, recover_n_FIAD, FPR_FIAD_tmp, FNR_FIAD_tmp, F1_FIAD_tmp, Precision_FIAD_tmp, Recall_FIAD_tmp = freq_itemset_mining_HST(support_list, n, e, ratio)
            elapsed_time = execute_with_func_timeout(freq_itemset_mining_HST, timeout_seconds=3600, Amplify_Support_matrix=support_list, n=n, e=e, ratio=ratio)

        else:
            print('No valid Protocol!==============')
            exit(0)
        from Diffstats import diffstats
        time_diff_1 = time.time()
        recover_data_one, recover_n_one, FPR_one_num_tmp, FNR_one_num_tmp, F1_one_num_tmp, Precision_one_num_tmp, Recall_one_num_tmp = diffstats(epsilon, domain, n, one_list, support_list, ratio, file_name, target_set_size, fix_ratio, ESTIMATE_DIST_attack, perturb_method)
        time_diff_2 = time.time()


    print('FIAD time = ', elapsed_time)
    print('Diffstats time = ', time_diff_2 - time_diff_1)


    with open('Result_time' + '/' + file_name + '/' + 'results_'  + '_' + args.vswhat + '_e=' + str(e) + '_r=' + str(target_set_size) + '_b=' + str(ratio) + '_s=' + str(splits) + '_h='  + str(h_ao) + '_diffstats_detect_'+ str(protocol) +'.txt', 'w') as file:
        file.write(f'FIAD time = {(elapsed_time)}\n')
        file.write(f'Diffstats runtime = {(time_diff_2 - time_diff_1)}\n')



def vs_epsilon(main_func):
    global g
    for e in [0.1, 0.5, 1]:
        print(e, end=' ')
        args.epsilon = float(e)
        ratio = 0.05
        # OLH
        # g-> hash function projection domain d' = e^epsilon + 1
        g = int(round(math.exp(args.epsilon))) + 1
        print(g, end=' ')
        main_func(e, target_set_size)

def vs_epsilon_1(main_func):
    global g
    for e in [1]:
        print(e, end=' ')
        args.epsilon = float(e)
        # OLH
        # g-> hash function projection domain d' = e^epsilon + 1
        g = int(round(math.exp(args.epsilon))) + 1
        print(g, end=' ')
        main_func(e, target_set_size)

def vs_beta(main_func):
    global ratio, g
    for ratio in [0.01, 0.05, 0.1]:
        e = 1
        print(e, end=' ')
        args.epsilon = float(e)
        # OLH
        # g-> hash function projection domain d' = e^epsilon + 1
        g = int(round(math.exp(args.epsilon))) + 1
        print(g, end=' ')
        main_func(e, target_set_size)

def vs_r(main_func):
    global g
    for target_set_size in [1, 5, 10]:
        for e in [1]:
            print(e, end=' ')
            args.epsilon = float(e)
            args.splits = target_set_size
            g = int(round(math.exp(args.epsilon))) + 1
            print(g, end=' ')
            main_func(e, target_set_size)
def dispatcher(main_func):

    vswhat = args.vswhat
    if vswhat == 'epsilon':
       vs_epsilon(main_func)
    if vswhat == 'beta':
        vs_beta(main_func)
    if vswhat == 'r':
        vs_r(main_func)
    if vswhat == 'epsilon_1':
        vs_epsilon_1(main_func)

def arg_parse():

    parser = argparse.ArgumentParser(description='Comparisor of different schemes.')
    parser.add_argument('--exp_round', type=int, default=1,
                        help='specify the n_userations for the experiments, default 10')
    parser.add_argument('--epsilon', type=float, default=0,
                        help='specify the differential privacy parameter, epsilon')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='specify the ratio of the fake user')
    parser.add_argument('--splits', type=int, default=10,
                        help='specify the splits size in MGA-A attack')
    parser.add_argument('--h_ao', type=int, default=0,
                        help='APA adaptive one attack radius')
    parser.add_argument('--target_set_size', type=int, default=10,
                        help='specify the number of the target item')
    parser.add_argument('--vswhat', type=str, default='epsilon',
                        help='specify the experiment setting')
    parser.add_argument('--dataset', type=str, default='emoji',
                        help='specify the datasets')
    parser.add_argument('--protocol', type=str, default='HST_User',
                        help='specify the protocol')

    return parser.parse_args()

if __name__ == '__main__':

    args = arg_parse()
    ratio = args.ratio
    file_name = args.dataset
    target_set_size = args.target_set_size
    protocol = args.protocol
    post_method = None

    dispatcher(main_OAP)


