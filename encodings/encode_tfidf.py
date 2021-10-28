# !pip install nltk
# !pip install spacy
# !python3 -m spacy download en_core_web_sm

import argparse
import numpy as np
import os
import string

import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import spacy
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

parser = argparse.ArgumentParser(description='tfidf Text Encoder')
parser.add_argument("--data_set", type=str, default='imdb')

splits = 5

def tokenizeText(text):
    
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    
    tokens = nlp(text)
    
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    
    tokens = [tok for tok in tokens if tok.lower() not in stoplist]
    
    tokens = [tok for tok in tokens if tok not in symbols]
    tokens = [tok for tok in tokens if len(tok) >= 3]
    tokens = [tok for tok in tokens if tok.isalpha()]
    
    tokens = list(set(tokens)) # bag of words
    #tokens = list((tokens))
    
    return ' '.join(tokens[:])

if __name__ == "__main__":
    args = parser.parse_args()
    data_set = args.data_set
    print('Starting to embedd %s data set ...' % data_set)

    if True:
    #if not (os.path.isfile('./%s_X_train.npy') and os.path.isfile('./%s_X_test.npy') and os.path.isfile('./%s_y_train') and os.path.isfile('./%s_y_test.npy')): 

        print('Load %s data set ...' % data_set)
        data_X = np.load('./../data/datasets/%s_X.npy' % data_set)

        print('Tokenize dataset ...')
        nlp = spacy.load("en_core_web_sm")
        stoplist = set(stopwords.words('english'))
        symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”","''"]
        
        X_tokenizes = [tokenizeText(x) for x in (data_X)]
        X_tokenizes = np.array(X_tokenizes)
        
        print('Create-Train-Test-Split ...')
        sss = StratifiedShuffleSplit(n_splits=splits, test_size=0.5, random_state=0)
        
        data_y = np.load('./../data/datasets/%s_y.npy' % data_set)
        
        sss.get_n_splits(X_tokenizes, data_y)
        i = 0
        for train_index, test_index in sss.split(X_tokenizes, data_y):
            
            X_train, X_test = X_tokenizes[train_index], X_tokenizes[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
        
            print(f'Encode ... #{i}')
            count_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=20000)
            count_vect.fit(X_train)
            
            X_train = count_vect.transform(X_train)
            X_test = count_vect.transform(X_test)
        
            print(f'Save split ... #{i}')
            np.save('./enc/%s_tfidf%s_X_train' % (data_set, i), X_train)
            np.save('./enc/%s_tfidf_%s_X_test' % (data_set, i), X_test)    
            np.save('./enc/%s_tfidf_%s_y_train' % (data_set, i), y_train)    
            np.save('./enc/%s_tfidf_%s_y_test' % (data_set, i), y_test)    
            
            i = i + 1
        print('Done')
    else:
        print('Encoding already exists.')