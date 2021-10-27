# https://aclanthology.org/C08-1143.pdf

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np

k = 20

class DensityEstimator:
    
    def __init__(self, X_pool):
        self.X_pool = X_pool
        self.density_scores = self._calc_density()
    
    def _calc_density(self):
        print('calc density ...')
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='cosine')
        nn.fit(self.X_pool)
    
        distances, indices = nn.kneighbors(self.X_pool)
        density_scores = []
        for j in tqdm(range(len(self.X_pool))):
            ds = []
            for i in range(1, len(indices[j])):
                cos = (1 - cosine(self.X_pool[indices[j][0]], self.X_pool[indices[j][i]]))
                ds.append(cos)
            ds = np.array(ds)
            density_score = ds.mean(axis=0)
            density_scores.append(density_score)
    
        return np.array(density_scores)
    
    def density_score(self, X_pool_index):
        return np.array([self.density_scores[i] for i in X_pool_index])
    