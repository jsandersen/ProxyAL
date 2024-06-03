import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/../.')

from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from timeit import default_timer as timer
from collections import Counter

import scipy
from enum import Enum, auto
import numpy as np

from src.al_seed import init
from src.DensityEstimator import DensityEstimator
from src.InstabilityEstimator import InstabilityEstimator
from src.UncertaintyEstimator import entropy, smallest_margin

from abc import ABC, abstractmethod

step_size = 10

class QueryStrategy(Enum):
    unc = auto()
    unc_density = auto()
    unc_isls = auto()
    isli = auto()
    isli_density = auto()
    isls = auto()
    isls_density = auto()
    
    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return QueryStrategy[s]
        except KeyError:
            raise ValueError()

class ActiveLearner(ABC):
    
    def __init__(self, warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, selection):
        self.warmstart_size = warmstart_size
        self.selection = selection
        
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.X_test = X_test
        self.y_test = y_test
        self.X_pool_index = X_pool_index
        
        self.X_al_training = []
        self.y_al_training = []
        self.X_al_index = []
    
        self.f1_mic_scores = []
        self.f1_mac_scores = []
        self.precision_score = []
        self.recall_scores = []
        self.c_scores = []
        self.times_train = []
        self.times_inf = []
        self.times_query = []
        self.n_train = 0
        
        if selection == QueryStrategy.unc_density or selection == QueryStrategy.isli_density or selection == QueryStrategy.isls_density:
            self.de = DensityEstimator(self.X_pool)
            
        if selection == QueryStrategy.isli_density or selection == QueryStrategy.isls_density or selection == QueryStrategy.isli or selection == QueryStrategy.isls or self.selection == QueryStrategy.unc_isls:
            self.ie = InstabilityEstimator()
        
        if isinstance(y_pool[0], int):
            self.classes = list(set(y_pool))
        else:
            self.classes = list(set(y_pool.flatten()))    
    
    def run_warmstart(self):
        query_start = timer()
        seed_idx = init(self.warmstart_size, self.X_pool, self.y_pool, self.classes) #k_means(self.warmstart_size, self.X_pool)
        query_end = timer()
        query_train = query_end - query_start
        
        # remove seed examples from pool
        for i in -np.sort(-np.array(seed_idx)):
            self.X_al_training.append(self.X_pool[i])
            self.y_al_training.append(self.y_pool[i])
            self.X_al_index.append(self.X_pool_index[i])
            
            del self.X_pool[i]
            del self.y_pool[i]
            del self.X_pool_index[i]

        # shuffle
        self.X_al_training, self.y_al_training = shuffle(self.X_al_training, self.y_al_training, random_state=42)
        
        # train
        f1_mic, f1_mac, precision, recall, c, time_train, time_inf = self._train() 
        self.f1_mic_scores.append(f1_mic)
        self.f1_mac_scores.append(f1_mac)
        self.precision_score.append(precision)
        self.recall_scores.append(recall)
        self.c_scores.append(c)
        self.times_train.append(time_train)
        self.times_inf.append(time_inf)
        self.times_query.append(query_train)
    
    def run_active_learning_rnd(self, n, step_size=10):
        print('rnd')
        for it in range(n):
            self.X_pool, self.y_pool, self.X_pool_index = shuffle(self.X_pool, self.y_pool, self.X_pool_index, random_state=42)
            
            # check
            self.X_al_training.extend(self.X_pool[:step_size])
            self.y_al_training.extend(self.y_pool[:step_size])
            self.X_al_index.extend(self.X_pool_index[:step_size])
            
            del self.X_pool[:step_size]
            del self.y_pool[:step_size]
            del self.X_pool_index[:step_size]
            
            f1_mic, f1_mac, precision, recall, c, time_train, time_inf = self._train()
            
            self.f1_mic_scores.append(f1_mic)
            self.f1_mac_scores.append(f1_mac)
            self.precision_score.append(precision)
            self.recall_scores.append(recall)
            self.c_scores.append(c)
            self.times_train.append(time_train)
            self.times_inf.append(time_inf)
            self.times_query.append(-1)
        
        return self.f1_mic_scores, self.f1_mac_scores, self.precision_score, self.recall_scores, self.c_scores, self.times_train, self.times_inf, self.times_query
    
    def run_active_learning(self, n, step_size=10):
        print('normal')
        for it in range(n):
            
            query_start = timer()
            index = self._query(self.X_pool, step_size)
            query_end = timer()
            query_train = query_end - query_start
            
            for i in index:
                self.X_al_training.append(self.X_pool[i])
                self.y_al_training.append(self.y_pool[i])
                self.X_al_index.append(self.X_pool_index[i])

            index[::-1].sort()
            for i in index:    
                del self.X_pool[i]
                del self.y_pool[i]
                del self.X_pool_index[i]
            
            f1_mic, f1_mac, precision, recall, c, time_train, time_inf = self._train()
            
            self.f1_mic_scores.append(f1_mic)
            self.f1_mac_scores.append(f1_mac)
            self.precision_score.append(precision)
            self.recall_scores.append(recall)
            self.c_scores.append(c)
            self.times_train.append(time_train)
            self.times_inf.append(time_inf)
            self.times_query.append(query_train)
        
        return self.f1_mic_scores, self.f1_mac_scores, self.precision_score, self.recall_scores, self.c_scores, self.times_train, self.times_inf, self.times_query
    
    @abstractmethod
    def _query(self, X_pool):
        while False:
            yield None
    
    def get_training_index(self):
        return self.X_al_index
    
    def get_data_used(self, X, y):
        X = [X[i] for i in self.X_al_index]
        y = [y[i] for i in self.X_al_index]
        return X, y
        
    def _print_metrics(self, y_pred, time_train, time_inf):
        f1_mic = round(f1_score(self.y_test, y_pred, average='micro'), 5)
        f1_mac = round(f1_score(self.y_test, y_pred, average='macro'), 5)
        precision = -1 # not used
        recall = -1 #not used
        c = Counter(self.y_al_training) 
        #time = time
        
        print('#%s  f1_mic: %s, f1_mac: %s, prec: %s, rec: %s, n: %s, c : %s, t_train: %s, t_inf: %s' % (self.n_train, f1_mic, f1_mac, precision, recall, len(self.X_al_training), c.most_common(), round(time_train, 4), round(time_inf, 4)))
        return f1_mic, f1_mac, precision, recall, c


##########################################
##########################################
##########################################

class SKLearnActiveLearner(ActiveLearner):
    
    def __init__(self, warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, Model, model_args = {}, selection = QueryStrategy.unc):
        self.Model = Model
        self.model_args = model_args
        super().__init__(warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, selection)
        
    def _train(self):
        self.n_train = self.n_train + 1
        self.clf = self.Model(**self.model_args)
        
        # training
        train_start = timer()
        self.clf.fit(self.X_al_training, self.y_al_training)
        train_end = timer()
        time_train = train_end - train_start
        
        # inference
        inf_start = timer()
        proba = self.clf.predict_proba(self.X_test)
        y_pred = proba.argmax(axis=1)
        inf_end = timer()
        time_inf = inf_end - inf_start
        
        # log
        f1_mic, f1_mac, precision, recall, c = self._print_metrics(y_pred, time_train, time_inf)
        return f1_mic, f1_mac, precision, recall, c, time_train, time_inf
     
    def _query(self, X_pool, step_size):
        proba_pool = self.clf.predict_proba(X_pool)
        ranking = smallest_margin(proba_pool) #entropy(proba_pool) 
        
        if self.selection == QueryStrategy.unc_density:
            ds = self.de.density_score(self.X_pool_index)
            ranking = ds * ranking
        
        if self.selection == QueryStrategy.isli_density:            
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            isli = self.ie.isli()
            ds = self.de.density_score(self.X_pool_index)
            ranking = ds * isli
            
        if self.selection == QueryStrategy.isli:            
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            isli = self.ie.isli()
            ranking =  isli
            
        if self.selection == QueryStrategy.isls_density:
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            isls = self.ie.isls()
            ds = self.de.density_score(self.X_pool_index)
            ranking = ds * isls
            
        if self.selection == QueryStrategy.isls:
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            isls = self.ie.isls()
            ranking = isls  
            
        if self.selection == QueryStrategy.unc_isls:
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            islsq = self.ie.islsq()
            ranking = ranking * islsq    
   
        return (-ranking).argsort()[:step_size]

    
##########################################
##########################################
##########################################

class SKLearnActiveLearnerSparse(ActiveLearner):
    
    def __init__(self, warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, Model, model_args = {}, selection = QueryStrategy.unc):
        self.Model = Model
        self.model_args = model_args
        super().__init__(warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, selection)
        
    def _train(self):
        self.n_train = self.n_train + 1
        X = scipy.sparse.vstack(self.X_al_training)
        self.clf = self.Model(**self.model_args)
        
        # training
        train_start = timer()
        self.clf.fit(X, self.y_al_training)
        train_end = timer()
        time_train = train_end - train_start
        
        # inference
        inf_start = timer()
        proba = self.clf.predict_proba(self.X_test)
        y_pred = proba.argmax(axis=1)
        inf_end = timer()
        time_inf = inf_end - inf_start
        
        f1_mic, f1_mac, precision, recall, c = self._print_metrics(y_pred, time_train, time_inf)
        return f1_mic, f1_mac, precision, recall, c, time_train, time_inf
     
    def _query(self, X_pool, step_size):
        X = scipy.sparse.vstack(self.X_al_training)
        proba_pool = self.clf.predict_proba(X)
        unc = entropy(proba_pool)
       
        return (-unc).argsort()[:step_size]

    
##########################################
##########################################
##########################################
