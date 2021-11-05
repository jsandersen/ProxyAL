from sklearn.utils import shuffle
import pandas as pd
import re
import numpy as np
import os
import unicodedata
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm 
import nltk
from nltk.corpus import reuters 

cat_8 = ['acq', 'earn', 'grain', 'interest', 'money-fx']

def _load_reuters(data_dir = None):
    nltk.download('reuters')
    
    dic_map = {}
    for i in range(len(cat_8)):
        dic_map[cat_8[i]] = i

    documents = reuters.fileids()
    train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
    test_docs = list(filter(lambda doc: doc.startswith("test"), documents));
    
    def _get_data(docs):
        docs_text = []
        docs_label = []
        for index, i in  enumerate(docs):
            docs_text.append(reuters.raw(fileids=[i]))
            docs_label.append(reuters.categories(i))

        X = []
        y = []
        for cat in cat_8:
            count = 0
            for i in range(len(docs_label)): 
                if cat in docs_label[i]:
                    if len(docs_label[i]) > 1:
                        double = 0
                        for label in docs_label[i]:
                            if label in cat_8:
                                double += 1
                        if double > 1:
                            continue;
                    count += 1
                    X.append(docs_text[i])
                    y.append(cat)
        return X, y
    
    X_train, y_train = _get_data(train_docs)
    X_test, y_test = _get_data(test_docs)
    
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    y = [dic_map[i] for i in y]
    
    return X, y

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_email(text):
    url = re.compile(r'([A-Za-z0-9]*\.)*[A-Za-z0-9]*@([A-Za-z]*\.?[A-Za-z0-9]*)*')
    return url.sub(r'',text)

if __name__ == "__main__":
    print('Prepare Reuters dataset ...')
    
    if True:
    #if not (os.path.isfile('./datasets/news_groups_X.npy') and os.path.isfile('./datasets/news_groups_X.npy')): 
        X, y = _load_reuters()
        X, y = shuffle(X, y, random_state = 42)
        
        for i in tqdm(range(len(X))):
            for character in ["\n", "|", ">", "<", "-", "+", "^", "[", "]", "#", "\t", "\r", "`", '_']:
                X[i] = X[i].replace(character, " ")
            X[i] = remove_URL(X[i])
            X[i] = remove_email(X[i])
        
        
        
        np.save('./datasets/reuters_X', X)
        np.save('./datasets/reuters_y', y)
        
        
        print('Done')
    else:
        print('Reuters dataset already exists.')
    