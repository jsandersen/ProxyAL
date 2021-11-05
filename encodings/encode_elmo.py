# pip install -U sentence-transformers
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import argparse
import numpy as np
import os

from tqdm import tqdm

import tensorflow_hub as hub


from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

parser = argparse.ArgumentParser(description='SBERT Text Encoder')
parser.add_argument("--data_set", type=str, default='imdb')

splits = 5

# ELMo Embedding
def elmo_default_fast(x, elmo):
    g = tf.compat.v1.Graph()
    with g.as_default():
        text_input = tf.compat.v1.placeholder(dtype=tf.compat.v1.string, shape=[None])
        embed = elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        embedded_text = embed(text_input)
        init_op = tf.compat.v1.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    g.finalize()

    # Create session and initialize.
    session = tf.compat.v1.Session(graph=g)
    session.run(init_op)

    embeddings = []
    for i in tqdm(range(0, x.shape[0],1)):
        embed = session.run(embedded_text, feed_dict={text_input: x[i:i+1]})
        embeddings.extend(embed)
    
    return np.array(embeddings)
    #return session.run(embedded_text, feed_dict={text_input: x})


# ELMo Embedding
def elmo_default(x, elmo):
    embeddings = elmo(x, signature="default", as_dict=True)["default"]

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        return sess.run(embeddings).tolist()

if __name__ == "__main__":
    args = parser.parse_args()
    data_set = args.data_set
    print('Starting to embedd %s data set ...' % data_set)
    
    if True:
    #if not (os.path.isfile('./%s_sbert_4_X_train.npy' % data_set) and os.path.isfile('./%s_sbert_4_X_test.npy' % data_set) and os.path.isfile('./%s_sbert_4_y_train' % data_set) and os.path.isfile('./%s_sbert_4_X_test.npy' % data_set)): 

        print('Load %s data set ...' % data_set)
        data_X = np.load('./../data/datasets/%s_X.npy' % data_set)

        print('Encode ...')
        
        embeddings = elmo_default_fast(data_X, None)
    
        
        #for i in tqdm(range(0,data_X.shape[0],25)):
        #    embed = elmo_default(data_X[i:i+25], elmo)
        #    np.save(f'./tmp/elmo_{data_set}_{i}_{i+25}', embed)
        
        #data_X_list = [data_X[i:i+20] for i in range(0,data_X.shape[0],20)]
        #embeddings = np.array([elmo_default(x, elmo) for x in tqdm(data_X_list)])
        #data_X = None
        
        #ex = []
        #for i in range(len(embeddings)):
        #    ex.extend(embeddings[i])
        #embeddings = np.array(ex)
        
        print('Create-Train-Test-Split ...')
        sss = StratifiedShuffleSplit(n_splits=splits, test_size=0.5, random_state=0)
        
        data_y = np.load('./../data/datasets/%s_y.npy' % data_set)
       
        
        sss.get_n_splits(embeddings, data_y)
        i = 0
        for train_index, test_index in sss.split(embeddings, data_y):
            X_train, X_test = embeddings[train_index], embeddings[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
        
            print('Save split ...')
            np.save('./enc/%s_elmo_%s_X_train' % (data_set, i), X_train)
            np.save('./enc/%s_elmo_%s_X_test' % (data_set, i), X_test)    
            np.save('./enc/%s_elmo_%s_y_train' % (data_set, i), y_train)    
            np.save('./enc/%s_elmo_%s_y_test' % (data_set, i), y_test)    
            
            i = i + 1
        print('Done')
    else:
        print('Encoding already exists.')