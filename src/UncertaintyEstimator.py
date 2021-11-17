from scipy.stats import entropy as entropy2
import numpy as np

np.random.seed(42)

def least_confidence(probs):
    return 1 - np.array(probs).max(axis=1)

def entropy(probs):
    return entropy2(probs, base=2, axis=1)

def smallest_margin(probs):
    part = np.partition(-probs, 1, axis=1)
    return part[:, 0] - part[:, 1]
    
def random(probs):
    return np.random.rand(len(probs))
    