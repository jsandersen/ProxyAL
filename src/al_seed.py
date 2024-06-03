import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

def init(k, X_pool, y_pool, classes):
    
    init = []
    for c in classes:
        count = 0    
        for i in range(len(y_pool)):
            if count < k:
                if y_pool[i] == c:
                    init.append(i)                    
                    count = count + 1
            else:
                break;
    return init