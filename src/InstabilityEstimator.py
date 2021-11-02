# https://dl.acm.org/doi/pdf/10.1145/2093153.2093154?casa_token=jfJEiAOiXLcAAAAA:hlZFoIHLa1yQ_B_vLUNv7JT99RYcn0lPK0UCJtHdQXsC9gZEgG1H-I1jYDY1DREDEHzkTTREmrx0

import numpy as np

n = 5

class InstabilityEstimator:
    
    def __init__(self):
        self.index_history = []
        self.unc_history = []
        self.preds_class_history = []

    def next_iteration(self, index, unc, preds_class):
        # limit history
        self.index_history[-n:]
        self.unc_history[-n:]
        self.preds_class_history[-n:]        
        
        unc = np.array(unc)
        preds_class = np.array(preds_class)
        index = np.array(index)
        
        self.index_history.append(index)
        self.preds_class_history.append(preds_class)
        self.unc_history.append(np.array(unc))
        for j in range(len(self.unc_history)):
            diff = np.in1d(np.array(self.index_history[j]), index, assume_unique=True)
            self.index_history[j] = self.index_history[j][diff]
            self.unc_history[j] = self.unc_history[j][diff]
            self.preds_class_history[j] = self.preds_class_history[j][diff]
    
    def isli(self):
        preds = self.unc_history
        l = len(preds) if n > len(preds) else n
        i = len(preds)
        _sum = 0
        for k in range(i-l, i):
            _sum = _sum + (preds[k] - preds[k-1])
        return np.array(preds[i-1] + _sum)
    
    def isls(self):
        preds = self.unc_history
        preds_class = self.preds_class_history
        l = len(preds) if n > len(preds) else n
        i = len(preds)
        _sum = 0
        for k in range(i-l, i):
            _sum = _sum + ( (preds_class[k] != preds_class[k-1]) * (preds[k] - preds[k-1]) )
        return np.array(preds[i-1] + _sum)
    
    def islsq(self):
        preds = self.unc_history
        preds_class = self.preds_class_history
        l = len(preds) if n > len(preds) else n
        i = len(preds)
        _sum = 0
        for k in range(i-l, i):
            _sum = _sum + ( (preds_class[k] == preds_class[k-1]) * (preds[k] - preds[k-1]) )
        return np.array(preds[i-1] + _sum)