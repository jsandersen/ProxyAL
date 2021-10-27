import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

def k_means_balance(k, X_pool, y_pool, classes):
    
    #init = []
    #for c in classes:
    #    for i in range(len(y_pool)):
    #        if y_pool[i] == c:
    #            init.append(i)
    #            break;      
    # 
    #print('k-means seeding ...')
    #kmeans = KMeans(init='k-means++', n_clusters=k, random_state=0)
    #kmeans.fit(X_pool)
    #closest, _ = vq(kmeans.cluster_centers_, X_pool)
    #return [*closest, *init]

    
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
        


        #for c in self.classes:
        #    count = 0    
            # add x of each class
            #for i in range(len(self.y_pool)):
            #    if count < self.warmstart_size:
            #        if self.y_pool[i] == c:
            #            self.X_al_training.append(self.X_pool[i])
            #            self.y_al_training.append(self.y_pool[i])
            #            self.X_al_index.append(self.X_pool_index[i])
            #            
            #            del self.X_pool[i]
            #            del self.y_pool[i]
            #            del self.X_pool_index[i]
            #            
            #            count = count + 1
            #    else:
            #        break;    