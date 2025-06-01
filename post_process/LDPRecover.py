import numpy as np
import math

def LDP_Recover(Estimation, eta, epsilon, domain, n):
    '''
    The code of LDP_Recover based on
    LDPRecover: Recovering Frequencies from Poisoning Attacks against Local Differential Privacy
    :param Estimation: The Estiamtion of LDP including malicious frequencies
    :param eta: The ratio of the number of malicious users to the number of genuine users
                eta = m / n
    :param n: number of user
    :return: Recovered Frequencies.
    '''
    g = int(round(math.exp(epsilon))) + 1
    p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
    q = 1 / g

    estimates = np.copy(Estimation)
    estimates = estimates / n
    # D_0 in paper
    mask0 = estimates <= 0
    # D_1
    mask1 = estimates > 0
    # malicious frequencies
    f_y_prime = 1 / len(mask1) * (1 - q * domain) / (p - q)
    # estimated genuine frequencies
    f_x_tilde = (1 + eta) * estimates - eta * f_y_prime
    results = Refining_Recovered_Frequencies(f_x_tilde, domain)
    return results * n


def Refining_Recovered_Frequencies(f_x_tilde, domain):
    '''
    Using KKT condition to refine the recovered frequency
    :param f_x_tilde: not consistent genuine frequencies
    :param domain: domain
    :return: Refined genuine frequencies
    '''
    # initialize the D_star with full domain
    D_s_l = [i for i in range(domain)]
    D_s_u = []
    # initialize f_x_prime(refined genuine frequencies)
    f_x_prime = f_x_tilde - 1 / len(D_s_l) * (sum([f_x_tilde[i] for i in D_s_l]) - 1)
    while np.min(f_x_prime[D_s_l]) < 0:
        for i in reversed(range(len(D_s_l))):
            if f_x_prime[D_s_l[i]] < 0:
                D_s_u.append(D_s_l[i])
                del D_s_l[i]
        f_x_prime[D_s_l] = f_x_tilde[D_s_l] - 1 / len(D_s_l) * (sum([f_x_tilde[i] for i in D_s_l]) - 1)
    f_x_prime[D_s_u] = 0
    return f_x_prime



'''Estimation = []
eta = 0.05
epsilon = 5
domain = 30
results = LDP_Recover(Estimation, eta, epsilon, domain)
print(results)'''