# !pip install nltk
# !pip install spacy
# !python3 -m spacy download en_core_web_sm

import argparse
import numpy as np
import os
import string
import pickle

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
    
    tokens = list((tokens))
    
    return ' '.join(tokens[:])

import wordfreq

def sentence_vector_avg_W(sentence, glove, a=1e-3):
    """Calculate the sentence vector as the mean of the word vectors"""
    
    word_vecs = []
    for token in sentence:
        token = token.lower()
        try:
            vw = np.array(glove[token])
            freq = wordfreq.word_frequency(token, 'en', wordlist='large')
            vw *= a / (a + freq)
            word_vecs.append(vw)
        except KeyError:
            pass
        
    
    if len(word_vecs) < 1:
        return np.zeros(300)
    else:
        return np.array(word_vecs).mean(axis=0)

def embed_avg_W(sentences, glove):
    """Calculate sentence embeddings for set of sentences
    based on the average of the word vectors
    """
    
    values = []
    for s in sentences:
        mean = sentence_vector_avg_W(s, glove)
        values.append(mean)
        
    return np.array(values)


from sklearn.decomposition import PCA, TruncatedSVD

def embed_avg_WR(sentences, glove):  
    """Calculate sentence embeddings for set of sentences
    based on the weighted average of the word vectors +
    applying subtraction of first principal component projection.
    """
    
    # weighted average of word vectors for all sentences
    values = []
    for s in sentences:
        mean = sentence_vector_avg_W(s, glove)  # get_weighted_average
        values.append(mean)
        
    X = np.array(values)
    
    # removal of common component
    #pca = PCA(n_components=1).fit(X)
    #u = pca.components_#[0]
    
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0).fit(X) # compute_pc
    u = svd.components_[0] # pc
    
    return X - np.dot(X, np.outer(u, u)) # pc

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
        
        with open('glove.840B.300d.pkl', 'rb') as pkl:
            glove = pickle.load(pkl)
        
        print(data_X[0])
        
        X_tokenizes = [tokenizeText(x) for x in (data_X)]
        X_tokenizes = [x.split(' ') for x in X_tokenizes]
        
        print(X_tokenizes[0])
        
        np.save('test', X_tokenizes)
        
        X_embed = embed_avg_WR(X_tokenizes, glove)
        
        print('Create-Train-Test-Split ...')
        sss = StratifiedShuffleSplit(n_splits=splits, test_size=0.5, random_state=0)
        
        data_y = np.load('./../data/datasets/%s_y.npy' % data_set)
        
        sss.get_n_splits(X_embed, data_y)
        i = 0
        for train_index, test_index in sss.split(X_embed, data_y):
            
            X_train, X_test = X_embed[train_index], X_embed[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
        
            print(f'Save split ... #{i}')
            np.save('./enc/%s_sif%s_X_train' % (data_set, i), X_train)
            np.save('./enc/%s_sif_%s_X_test' % (data_set, i), X_test)    
            np.save('./enc/%s_sif_%s_y_train' % (data_set, i), y_train)    
            np.save('./enc/%s_sif_%s_y_test' % (data_set, i), y_test)    
            
            i = i + 1
        print('Done')
    else:
        print('Encoding already exists.')