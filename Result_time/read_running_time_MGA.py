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
                h_ao) + '_diffstats_detect_' + str(perturb_method) + '.txt', 'r') as file:
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
        with open(file_name + '/' + 'results_' + '_' + vswhat + '_e=' + str(1) + '_r=' + str(
                r) + '_b=' + str(iter) + '_s=' + str(splits) + '_h=' + str(
                h_ao) + '_diffstats_detect_' + str(perturb_method) + '.txt', 'r') as file:
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
        # vsepsilon
        with open(file_name + '/' + 'results_' + '_' + vswhat + '_e=' + str(1) + '_r=' + str(
                iter) + '_b=' + str(ratio) + '_s=' + str(iter) + '_h=' + str(
                h_ao) + '_diffstats_detect_' + str(perturb_method) + '.txt', 'r') as file:
            for i, line in enumerate(file):
                if i in [0]:
                    numbers = number_regex.findall(line)
                    if numbers:
                        current_values.append(float(numbers[0]))

        results.append(current_values)
    print('result of {}: {}'.format(perturb_method, results))


file_name = 'zipf'
#epsilon
print('Time epsilon: ')
for protocol in ['OUE', 'OLH_User', 'OLH_Server', 'HST_User', 'HST_Server']:
    read_epsilon(10,file_name, 'epsilon', 0.05, 10, 0, protocol)
print('Time beta: ')
for protocol in ['OUE', 'OLH_User', 'OLH_Server', 'HST_User', 'HST_Server']:
    read_beta(10,file_name, 'beta', 0.05, 10, 0, protocol)

print('Time r: ')
for protocol in ['OUE', 'OLH_User', 'OLH_Server', 'HST_User', 'HST_Server']:
    read_r(10, file_name, 'r', 0.05, 10, 0, protocol)
