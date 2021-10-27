import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/../.') 

import numpy as np
from sklearn.linear_model import LogisticRegression
from src.training_loop2 import SKLearnActiveLearner, KerasActiveLearner
from matplotlib import pyplot as plt
import pandas as pd
import argparse

save_dir = './app_store_sbert'
train_X_dir = './../encodings/enc/app_store_sbert_%s_X_train.npy'
train_y_dir = './../encodings/enc/app_store_sbert_%s_y_train.npy'
test_X_dir = './../encodings/enc/app_store_sbert_%s_X_test.npy'
test_y_dir = './../encodings/enc/app_store_sbert_%s_y_test.npy'

n = 5
    
#save_and_repeat = 2
#al_steps = 30
#sample_size_per_step = 1    

def getXtrain(): return np.load(train_X_dir % i ).tolist()
def getytrain(): return np.load(train_y_dir % i).tolist()
def np_flatten(x, dtype=None): return np.array(x, dtype=dtype)

parser = argparse.ArgumentParser(description='Run Active Learning with SBERT Encoder')
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--save_repeat", type=int, default=1)

if __name__ == "__main__":
    
    args = parser.parse_args()
    al_steps = args.steps
    save_and_repeat = args.save_repeat
    sample_size_per_step = args.batch_size
    
    save_dir = './res/%s_%s_%s_%s' % (save_dir, sample_size_per_step, al_steps, save_and_repeat)
    print('Save into ', save_dir, ' ...')
    
    
    f1_mic_list = []
    f1_mac_list = []
    precision_list = []
    recall_list = []
    c_list = []
    times_train_list = []
    times_inf_list = []
    
    used_index_meta_list = []
    
    for i in range(n):
        
        print('Start Model %s ...' % i)
        
        # load data
        X_train = getXtrain()
        X_test = np.load(test_X_dir % i)
        y_train = getytrain()
        y_test = np.load(test_y_dir % i)

        train_idx = [i for i in range(len(X_train))]

        used_index_list = []
        
        al = SKLearnActiveLearner(10, getXtrain(), getytrain(), X_test, y_test, train_idx, LogisticRegression, {'random_state':0})
        
        al.run_warmstart()
        print('__ End Warmstart')
        
        for i in range(save_and_repeat):
            f1_mic, f1_mac, precision, recall, c, times_train, times_inf = al.run_active_learning_rnd(al_steps, sample_size_per_step)
            
            print('__ Checkpoint: Save Data')
            
            # get selected data
            used_index = al.get_training_index()
            
            used_index_list.append(used_index[:]) # deep copy
        
        f1_mic_list.append(f1_mic)
        f1_mac_list.append(f1_mac)
        precision_list.append(precision)
        recall_list.append(recall)
        c_list.append(c)
        times_train_list.append(times_train)
        times_inf_list.append(times_inf)
            
        used_index_meta_list.append(used_index_list)
        
        print('Done AL')
        print()

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    np.save('%s/f1_micro' % save_dir, f1_mic_list)
    np.save('%s/f1_macro' % save_dir, f1_mac_list)
    np.save('%s/class_distributions' % save_dir, c_list)
    np.save('%s/times_training' % save_dir, times_train_list)
    np.save('%s/times_inference' % save_dir, times_inf_list)
    np.save('%s/used_training_index' % save_dir, used_index_meta_list)