from sklearn.utils import shuffle
import pandas as pd
import re
import numpy as np
import os
import unicodedata
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm 

def _load_toxic(data_dir = './datasets/labeled_data.csv'):
    nRowsRead = None
    df0 = pd.read_csv(data_dir, delimiter=',', nrows = nRowsRead)
    df0.dataframeName = data_dir
    nRow, nCol = df0.shape
    c=df0['class']
    df0.rename(columns={'tweet' : 'text', 'class' : 'category'}, inplace=True)
    a=df0['text']
    b=df0['category'].map({0: 'hate_speech', 1: 'offensive_language', 2: 'neither'})
    df= pd.concat([a,b,c], axis=1)
    
    X = df['text'].values
    y = df['class'].map({0: 0, 1: 0, 2: 1}).values
    return X, y

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_email(text):
    url = re.compile(r'([A-Za-z0-9]*\.)*[A-Za-z0-9]*@([A-Za-z]*\.?[A-Za-z0-9]*)*')
    return url.sub(r'',text)

if __name__ == "__main__":
    print('Prepare Toxic dataset ...')
    
    if True:
    #if not (os.path.isfile('./datasets/news_groups_X.npy') and os.path.isfile('./datasets/news_groups_X.npy')): 
        X, y = _load_toxic()
        X, y = shuffle(X, y, random_state = 42)
        
        for i in tqdm(range(len(X))):
            for character in ["\n", "|", ">", "<", "-", "+", "^", "[", "]", "#", "\t", "\r", "`", '_']:
                X[i] = X[i].replace(character, " ")
            X[i] = remove_URL(X[i])
            X[i] = remove_email(X[i])
            if X[i] == None:
                X[i] == ''
            X[i] = re.sub(r'&#34', '', X[i])
            
        
        np.save('./datasets/toxic_X', X.tolist())
        np.save('./datasets/toxic_y', y.tolist())
        
        
        print('Done')
    else:
        print('Toxic dataset already exists.')
    