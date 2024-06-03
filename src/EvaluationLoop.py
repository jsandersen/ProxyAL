import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/../.') 

import numpy as np
from sklearn.linear_model import LogisticRegression
from src.training_loop2 import SKLearnActiveLearner, KerasActiveLearner
from src.UncertaintyEstimator import entropy, random
n = 5

class EvaluationLoop:
    
    def __init__(self, save_dir, dataset, encoding, sample_size_per_step, al_steps, save_and_repeat, query_strategy, warmstart = 10, random=False):
        self.train_X_dir = f'./../encodings/enc/{dataset}_{encoding}_%s_X_train.npy'
        self.train_y_dir = f'./../encodings/enc/{dataset}_{encoding}_%s_y_train.npy'
        self.test_X_dir = f'./../encodings/enc/{dataset}_{encoding}_%s_X_test.npy'
        self.test_y_dir = f'./../encodings/enc/{dataset}_{encoding}_%s_y_test.npy'
        self.sample_size_per_step = sample_size_per_step
        self.al_steps = al_steps
        self.save_and_repeat = save_and_repeat
        self.query_strategy = query_strategy
        self.random = random
        self.warmstart = warmstart
        
        self.save_dir = '%s_%s_%s_%s' % (save_dir, sample_size_per_step, al_steps, save_and_repeat)
        print('Save directory: ', self.save_dir)
        
    def _getXtrain(self, i): return np.load(self.train_X_dir % i ).tolist()
    def _getytrain(self, i): return np.load(self.train_y_dir % i).tolist()

    def run(self):
        f1_mic_list = []
        f1_mac_list = []
        precision_list = []
        recall_list = []
        c_list = []
        times_train_list = []
        times_inf_list = []
        used_index_meta_list = []
        times_query_list = []
        
        for i in range(n):
            print('Start model %s ...' % i)
            
            # load data
            X_train = self._getXtrain(i)
            X_test = np.load(self.test_X_dir % i)
            y_train = self._getytrain(i)
            y_test = np.load(self.test_y_dir % i)
            
            train_idx = [i for i in range(len(X_train))]
            used_index_list = []
            
            uncertainty_sampling=entropy
            if self.random:
                uncertainty_sampling=random
            
            al = SKLearnActiveLearner(
                self.warmstart, # 10 
                self._getXtrain(i), 
                self._getytrain(i), 
                X_test,
                y_test,
                train_idx, 
                LogisticRegression, 
                {'random_state':0}, 
                self.query_strategy
            )
            
            al.run_warmstart()
            print('__ End Warmstart')
            
            for i in range(self.save_and_repeat):
                
                if self.random: # TODO random
                    f1_mic, f1_mac, precision, recall, c, times_train, times_inf, times_query = al.run_active_learning_rnd(self.al_steps, self.sample_size_per_step)
                else:
                    f1_mic, f1_mac, precision, recall, c, times_train, times_inf, times_query = al.run_active_learning(self.al_steps, self.sample_size_per_step)
                
                print('__ Checkpoint: Save Data')
                
                # get selected data
                used_index = al.get_training_index()
                used_index_list.append(used_index[:])
        
            f1_mic_list.append(f1_mic)
            f1_mac_list.append(f1_mac)
            precision_list.append(precision)
            recall_list.append(recall)
            c_list.append(c)
            times_train_list.append(times_train)
            times_inf_list.append(times_inf)
            times_query_list.append(times_query)

            used_index_meta_list.append(used_index_list)

            print('Done')
            print()
            
        # end loop
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
    
        np.save('%s/f1_micro' % self.save_dir, f1_mic_list)
        np.save('%s/f1_macro' % self.save_dir, f1_mac_list)
        np.save('%s/class_distributions' % self.save_dir, c_list)
        np.save('%s/times_training' % self.save_dir, times_train_list)
        np.save('%s/times_inference' % self.save_dir, times_inf_list)
        np.save('%s/used_training_index' % self.save_dir, used_index_meta_list)
        np.save('%s/times_query' % self.save_dir, times_query_list)