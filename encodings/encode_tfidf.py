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

parser = argparse.ArgumentParser(description='tfidf Text Encoder')
parser.add_argument("--data_set", type=str, default='imdb')

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

    if not (os.path.isfile('./%s_X_train.npy') and os.path.isfile('./%s_X_test.npy') and os.path.isfile('./%s_y_train') and os.path.isfile('./%s_y_test.npy')): 

        print('Load %s data set ...' % data_set)
        data_X = np.load('./../data/%s_X.npy' % data_set)

        print('Tokenize dataset ...')
        nlp = spacy.load("en_core_web_sm")
        stoplist = set(stopwords.words('english'))
        symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”","''"]
        
        X_tokenizes = [tokenizeText(x) for x in (data_X)]

        print('Create train-test-split ...')
        data_y = np.load('./../data/%s_y.npy' % data_set)
        X_train, X_test, y_train, y_test = train_test_split(X_tokenizes, data_y, test_size=0.50, random_state=42)

        print('Save split ...')
        np.save('tfidf_%s_X_train' % data_set, X_train)
        np.save('tfidf_%s_X_test' % data_set, X_test)    
        np.save('tfidf_%s_y_train' % data_set, y_train)    
        np.save('tfidf_%s_y_test' % data_set, y_test)    

        print('Done')
    else:
        print('Encoding already exists.')