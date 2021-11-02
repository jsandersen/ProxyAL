# pip install -U sentence-transformers
import argparse
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser(description='SBERT Text Encoder')
parser.add_argument("--data_set", type=str, default='imdb')

splits = 5

def cls_pooling_model(transformer_model):
    input_ids = tf.keras.Input(shape=(768,), name='input_token', dtype='int32')
    attention_mask = tf.keras.Input(shape=(768,), name='masked_token', dtype='int32')
    
    embedding_layer = transformer_model([input_ids, attention_mask])[0]
    cls_token = embedding_layer[:,0,:] # cls token

    bert_encoder = tf.keras.models.Model(inputs = {"input_ids": input_ids, "attention_mask": attention_mask}, outputs = cls_token)
    return bert_encoder

def tokenize(sentences, max_length=768, padding='max_length'):
    return tokenizer(
        sentences,
        truncation=True,
        padding=padding,
        max_length=max_length,
        return_tensors="tf"
    )

if __name__ == "__main__":
    args = parser.parse_args()
    data_set = args.data_set
    print('Starting to embedd %s data set ...' % data_set)
    
    if True:
    #if not (os.path.isfile('./%s_sbert_4_X_train.npy' % data_set) and os.path.isfile('./%s_sbert_4_X_test.npy' % data_set) and os.path.isfile('./%s_sbert_4_y_train' % data_set) and os.path.isfile('./%s_sbert_4_X_test.npy' % data_set)): 

        print('Load %s data set ...' % data_set)
        data_X = np.load('./../data/datasets/%s_X.npy' % data_set)

        print('Load encoder ...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        transformer_model = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=False)

        bert_encoder = cls_pooling_model(transformer_model)

        print('Encode ...')
        embeddings = []
        for i in tqdm(range(0, len(data_X), 100)):
            X = dict(tokenize(data_X[i:i+100].tolist()))
            embed = bert_encoder(X).numpy().tolist()
            embeddings.extend(embed)
        embeddings = np.array(embeddings)
            
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
            np.save('./enc/%s_bert_%s_X_train' % (data_set, i), X_train)
            np.save('./enc/%s_bert_%s_X_test' % (data_set, i), X_test)    
            np.save('./enc/%s_bert_%s_y_train' % (data_set, i), y_train)    
            np.save('./enc/%s_bert_%s_y_test' % (data_set, i), y_test)    
            
            i = i + 1
        print('Done')
    else:
        print('Encoding already exists.')