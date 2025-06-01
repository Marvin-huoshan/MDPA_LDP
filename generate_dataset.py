import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    # Compute the unnormalized Zipf probability
    probabilities = 1 / top_1000_values ** a
    # Probability Normalization
    probabilities /= probabilities.sum()
    # According to the probability, random sampling n times.
    samples = np.random.choice(top_1000_values, size=n, p=probabilities)
    samples -= 1
    # Count the number of times each value occurs
    for i in range(n):
        REAL_DIST[samples[i]] += 1
    X = samples


def generate_emoji_dist():

    global X
    # Load precomputed emoji ID sequence (shape: (n,))
    data = np.load('datasets/emoji.npy')
    X = np.copy(data)
    # for each sampled emoji ID, increment its count in REAL_DIST
    for i in range(n):
        REAL_DIST[data[i]] += 1


def generate_fire_dist():

    global X
    # Read the 'Unit_ID' column as a pandas Series
    values = pd.read_csv("datasets/fire.csv")["Unit_ID"]
    # Fit a label encoder to map each unique Unit_ID to an integer [0..d-1]
    lf = LabelEncoder().fit(values)
    # Transform the original IDs to integer labels
    data = lf.transform(values)
    X = np.copy(data)
    # for each sampled label, increment its count in REAL_DIST
    for i in range(n):
        REAL_DIST[data[i]] += 1