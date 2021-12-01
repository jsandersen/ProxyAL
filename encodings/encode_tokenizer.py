# pip install -U sentence-transformers
import argparse
import numpy as np
import os

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import spacy
from spacy.lang.en import English
nlp = spacy.load("en_core_web_sm")
STOPLIST = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

parser = argparse.ArgumentParser(description='Tokenizer Text Encoder')
parser.add_argument("--data_set", type=str, default='imdb')

splits = 5

import string

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”","''"]

def tokenizeText(text):
    
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    
    tokens = nlp(text)
    
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    
    tokens = [tok for tok in tokens if tok.lower() not in STOPLIST]
    
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    tokens = [tok for tok in tokens if len(tok) >= 3]
    tokens = [tok for tok in tokens if tok.isalpha()]
    
    #tokens = list(set(tokens))
    #tokens = list((tokens))
    
    return ' '.join(tokens[:])

if __name__ == "__main__":
    args = parser.parse_args()
    data_set = args.data_set
    print('Starting to embedd %s data set ...' % data_set)
    
    if True:
    #if not (os.path.isfile('./%s_sbert_4_X_train.npy' % data_set) and os.path.isfile('./%s_sbert_4_X_test.npy' % data_set) and os.path.isfile('./%s_sbert_4_y_train' % data_set) and os.path.isfile('./%s_sbert_4_X_test.npy' % data_set)): 

        print('Load %s data set ...' % data_set)
        data_X = np.load('./../data/datasets/%s_X.npy' % data_set)

        num_words = 5000

        from tensorflow.keras.preprocessing.text import Tokenizer
        tokenizer = Tokenizer(num_words=num_words)

        print('Encode ...')
        embeddings = sbert.encode(data_X)
        data_X = None

        print('Create-Train-Test-Split ...')
        sss = StratifiedShuffleSplit(n_splits=splits, test_size=0.5, random_state=0)
        
        data_y = np.load('./../data/datasets/%s_y.npy' % data_set)
        #data_index = np.load('./../data/%s_index.npy' % data_set)
        
        sss.get_n_splits(embeddings, data_y)
        i = 0
        for train_index, test_index in sss.split(embeddings, data_y):
            X_train, X_test = embeddings[train_index], embeddings[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
        
            print('Save split ...')
            np.save('./enc/%s_sbert_%s_X_train' % (data_set, i), X_train)
            np.save('./enc/%s_sbert_%s_X_test' % (data_set, i), X_test)    
            np.save('./enc/%s_sbert_%s_y_train' % (data_set, i), y_train)    
            np.save('./enc/%s_sbert_%s_y_test' % (data_set, i), y_test)    
            
            i = i + 1
        print('Done')
    else:
        print('Encoding already exists.')