import re
import numpy as np
def read_epsilon(r, file_name, vswhat, ratio, splits, h_ao, perturb_method):
    results = []
    number_regex = re.compile(r'=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
    for iter in [0.1,0.5,1]:

        current_values = []
        # vsepsilon
        with open(file_name + '/' + 'results_' + '_' + vswhat + '_e=' + str(iter) + '_r=' + str(
                r) + '_b=' + str(ratio) + '_s=' + str(splits) + '_h=' + str(
                h_ao) + '_ASD_detect_' + str(perturb_method) + '.txt', 'r') as file:
           for i, line in enumerate(file):
                if i in [0]:

                    numbers = number_regex.findall(line)
                    if numbers:

                        current_values.append(float(numbers[0]))
        results.append(current_values)
    print('result of {}: {}'.format(perturb_method, results))

def read_beta(r, file_name, vswhat, ratio, splits, h_ao, perturb_method):
    results = []

    number_regex = re.compile(r'=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
    for iter in [0.01, 0.05,0.1]:

        current_values = []
        # vsepsilon
        with open(file_name + '/' + 'results_' + '_' + vswhat + '_e=' + str(0.5) + '_r=' + str(
                r) + '_b=' + str(iter) + '_s=' + str(splits) + '_h=' + str(
                h_ao) + '_ASD_detect_' + str(perturb_method) + '.txt', 'r') as file:
            for i, line in enumerate(file):
                if i in [0]:

                    numbers = number_regex.findall(line)
                    if numbers:

                        current_values.append(float(numbers[0]))

        results.append(current_values)
    print('result of {}: {}'.format(perturb_method, results))

def read_r(r, file_name, vswhat, ratio, splits, h_ao, perturb_method):
    results = []

    number_regex = re.compile(r'=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
    for iter in [1,5,10]:

        current_values = []

        with open(file_name + '/' + 'results_' + '_' + vswhat + '_e=' + str(0.5) + '_r=' + str(
                iter) + '_b=' + str(ratio) + '_s=' + str(iter) + '_h=' + str(
                h_ao) + '_ASD_detect_' + str(perturb_method) + '.txt', 'r') as file:
            for i, line in enumerate(file):
                if i in [0]:
                    numbers = number_regex.findall(line)
                    if numbers:

                        current_values.append(float(numbers[0]))

        results.append(current_values)
    print('result of {}: {}'.format(perturb_method, results))

def read_r_prime(r, file_name, vswhat, ratio, splits, h_ao, perturb_method):
    results = []

    number_regex = re.compile(r'=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
    for iter in [2,6,8]:

        current_values = []

        with open(file_name + '/' + 'results_' + '_' + vswhat + '_e=' + str(0.5) + '_r=' + str(
                r) + '_b=' + str(ratio) + '_s=' + str(iter) + '_h=' + str(
                h_ao) + '_ASD_detect_' + str(perturb_method) + '.txt', 'r') as file:
            for i, line in enumerate(file):
                if i in [0]:
                    numbers = number_regex.findall(line)
                    if numbers:

                        current_values.append(float(numbers[0]))

        results.append(current_values)
    print('result of {}: {}'.format(perturb_method, results))

file_name = 'fire'
#epsilon
'''print('Time APA: ')
print('Time epsilon: ')
for protocol in ['OUE', 'OLH_User', 'HST_User']:
    read_epsilon(10,file_name, 'epsilon', 0.1, 4, 20, protocol)
print('Time beta: ')
for protocol in ['OUE', 'OLH_User', 'HST_User']:
    read_beta(10,file_name, 'beta', 0.1, 4, 20, protocol)
'''
print('Time GRR: ')
print('Time MGA: ')
for protocol in ['GRR']:
    read_epsilon(10, file_name, 'epsilon', 0.1, 10, 0, protocol)
    read_beta(10,file_name, 'beta', 0.1, 10, 0, protocol)
    read_r(10, file_name, 'r', 0.1, 10, 0, protocol)
print('Time MGA-A: ')
for protocol in ['GRR']:
    read_epsilon(10, file_name, 'epsilon', 0.1, 4, 0, protocol)
    read_beta(10,file_name, 'beta', 0.1, 4, 0, protocol)
    read_r_prime(10, file_name, 'r_prime', 0.1, 4, 0, protocol)

