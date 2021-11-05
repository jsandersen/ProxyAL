from sklearn.utils import shuffle
import pandas as pd
import re
import numpy as np
import os
import unicodedata
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm 

def _load_news_groups():
    newsgroups_docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=1)
    
    X = newsgroups_docs.data
    y = newsgroups_docs.target
    
    return X, y

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_email(text):
    url = re.compile(r'([A-Za-z0-9]*\.)*[A-Za-z0-9]*@([A-Za-z]*\.?[A-Za-z0-9]*)*')
    return url.sub(r'',text)

if __name__ == "__main__":
    print('Prepare 20NewsGroups dataset ...')
    
    if True:
    #if not (os.path.isfile('./datasets/news_groups_X.npy') and os.path.isfile('./datasets/news_groups_X.npy')): 
        X, y = _load_news_groups()
        X, y = shuffle(X, y, random_state = 42)
        
        for i in tqdm(range(len(X))):
            for character in ["\n", "|", ">", "<", "-", "+", "^", "[", "]", "#", "\t", "\r", "`", '_']:
                X[i] = X[i].replace(character, " ")
            X[i] = remove_URL(X[i])
            X[i] = remove_email(X[i])
        
        np.save('./datasets/news_groups_X', X)
        np.save('./datasets/news_groups_y', y)
        
        
        print('Done')
    else:
        print('20NewsGroups dataset already exists.')
    