import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/../.')

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.utils import shuffle
from timeit import default_timer as timer
from collections import Counter
from tensorflow.keras.utils import to_categorical
import tensorflow.python.keras.backend as K
from tensorflow import keras as tfk
import copy
import scipy
from gensim import models
from enum import Enum, auto

import pandas as pd
import numpy as np
from tqdm import tqdm  

from src.al_seed import k_means_balance
from src.DensityEstimator import DensityEstimator
from src.InstabilityEstimator import InstabilityEstimator
from src.UncertaintyEstimator import least_confidence, entropy, smallest_margin

from abc import ABC, abstractmethod

step_size = 10

class QueryStrategy(Enum):
    unc = auto()
    unc_density = auto()
    unc_isls = auto()
    isli = auto()
    isli_density = auto()
    isls = auto()
    isls_density = auto()
    
    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return QueryStrategy[s]
        except KeyError:
            raise ValueError()

class ActiveLearner(ABC):
    
    def __init__(self, warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, selection):
        self.warmstart_size = warmstart_size
        self.selection = selection
        
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.X_test = X_test
        self.y_test = y_test
        self.X_pool_index = X_pool_index
        
        self.X_al_training = []
        self.y_al_training = []
        self.X_al_index = []
    
        self.f1_mic_scores = []
        self.f1_mac_scores = []
        self.precision_score = []
        self.recall_scores = []
        self.c_scores = []
        self.times_train = []
        self.times_inf = []
        self.n_train = 0
        
        if selection == QueryStrategy.unc_density or selection == QueryStrategy.isli_density or selection == QueryStrategy.isls_density:
            self.de = DensityEstimator(self.X_pool)
            
        if selection == QueryStrategy.isli_density or selection == QueryStrategy.isls_density or selection == QueryStrategy.isli or selection == QueryStrategy.isls or self.selection == QueryStrategy.unc_isls:
            self.ie = InstabilityEstimator()
        
        if isinstance(y_pool[0], int):
            self.classes = list(set(y_pool))
        else:
            self.classes = list(set(y_pool.flatten()))    
    
    def run_warmstart(self):
        seed_idx = k_means_balance(self.warmstart_size, self.X_pool, self.y_pool, self.classes) #k_means(self.warmstart_size, self.X_pool)
        
        # remove seed examples from pool
        for i in -np.sort(-np.array(seed_idx)):
            self.X_al_training.append(self.X_pool[i])
            self.y_al_training.append(self.y_pool[i])
            self.X_al_index.append(self.X_pool_index[i])
            
            del self.X_pool[i]
            del self.y_pool[i]
            del self.X_pool_index[i]

        # shuffle
        self.X_al_training, self.y_al_training = shuffle(self.X_al_training, self.y_al_training, random_state=42)
        
        # train
        f1_mic, f1_mac, precision, recall, c, time_train, time_inf = self._train() 
        self.f1_mic_scores.append(f1_mic)
        self.f1_mac_scores.append(f1_mac)
        self.precision_score.append(precision)
        self.recall_scores.append(recall)
        self.c_scores.append(c)
        self.times_train.append(time_train)
        self.times_inf.append(time_inf)
    
    def run_active_learning_rnd(self, n, step_size=10):
        print('rnd')
        for it in range(n):
            self.X_pool, self.y_pool, self.X_pool_index = shuffle(self.X_pool, self.y_pool, self.X_pool_index, random_state=42)
            
            # check
            self.X_al_training.extend(self.X_pool[:step_size])
            self.y_al_training.extend(self.y_pool[:step_size])
            self.X_al_index.extend(self.X_pool_index[:step_size])
            
            del self.X_pool[:step_size]
            del self.y_pool[:step_size]
            del self.X_pool_index[:step_size]
            
            f1_mic, f1_mac, precision, recall, c, time_train, time_inf = self._train()
            
            self.f1_mic_scores.append(f1_mic)
            self.f1_mac_scores.append(f1_mac)
            self.precision_score.append(precision)
            self.recall_scores.append(recall)
            self.c_scores.append(c)
            self.times_train.append(time_train)
            self.times_inf.append(time_inf)
        
        return self.f1_mic_scores, self.f1_mac_scores, self.precision_score, self.recall_scores, self.c_scores, self.times_train, self.times_inf
    
    def run_active_learning(self, n, step_size=10):
        print('normal')
        for it in range(n):
            index = self._query(self.X_pool, step_size)
            
            for i in index:
                self.X_al_training.append(self.X_pool[i])
                self.y_al_training.append(self.y_pool[i])
                self.X_al_index.append(self.X_pool_index[i])

            index[::-1].sort()
            for i in index:    
                del self.X_pool[i]
                del self.y_pool[i]
                del self.X_pool_index[i]
            
            f1_mic, f1_mac, precision, recall, c, time_train, time_inf = self._train()
            
            self.f1_mic_scores.append(f1_mic)
            self.f1_mac_scores.append(f1_mac)
            self.precision_score.append(precision)
            self.recall_scores.append(recall)
            self.c_scores.append(c)
            self.times_train.append(time_train)
            self.times_inf.append(time_inf)
        
        return self.f1_mic_scores, self.f1_mac_scores, self.precision_score, self.recall_scores, self.c_scores, self.times_train, self.times_inf
    
    @abstractmethod
    def _query(self, X_pool):
        while False:
            yield None
    
    def get_training_index(self):
        return self.X_al_index
    
    def get_data_used(self, X, y):
        X = [X[i] for i in self.X_al_index]
        y = [y[i] for i in self.X_al_index]
        return X, y
        
    def _print_metrics(self, y_pred, time_train, time_inf):
        f1_mic = round(f1_score(self.y_test, y_pred, average='micro'), 5)
        f1_mac = round(f1_score(self.y_test, y_pred, average='macro'), 5)
        precision = -1 #round(precision_score(self.y_test, y_pred), 5)
        recall = -1 #round(recall_score(self.y_test, y_pred), 5)
        c = Counter(self.y_al_training) 
        #time = time
        
        print('#%s  f1_mic: %s, f1_mac: %s, prec: %s, rec: %s, n: %s, c : %s, t_train: %s, t_inf: %s' % (self.n_train, f1_mic, f1_mac, precision, recall, len(self.X_al_training), c.most_common(), round(time_train, 4), round(time_inf, 4)))
        return f1_mic, f1_mac, precision, recall, c


##########################################
##########################################
##########################################

class SKLearnActiveLearner(ActiveLearner):
    
    def __init__(self, warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, Model, model_args = {}, selection = QueryStrategy.unc):
        self.Model = Model
        self.model_args = model_args
        super().__init__(warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, selection)
        
    def _train(self):
        self.n_train = self.n_train + 1
        self.clf = self.Model(**self.model_args)
        
        # training
        train_start = timer()
        self.clf.fit(self.X_al_training, self.y_al_training)
        train_end = timer()
        time_train = train_end - train_start
        
        # inference
        inf_start = timer()
        proba = self.clf.predict_proba(self.X_test)
        y_pred = proba.argmax(axis=1)
        inf_end = timer()
        time_inf = inf_end - inf_start
        
        # log
        f1_mic, f1_mac, precision, recall, c = self._print_metrics(y_pred, time_train, time_inf)
        return f1_mic, f1_mac, precision, recall, c, time_train, time_inf
     
    def _query(self, X_pool, step_size):
        proba_pool = self.clf.predict_proba(X_pool)
        ranking = least_confidence(proba_pool) # sampling method []
        
        if self.selection == QueryStrategy.unc_density:
            ds = self.de.density_score(self.X_pool_index)
            ranking = ds * ranking
        
        if self.selection == QueryStrategy.isli_density:            
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            isli = self.ie.isli()
            ds = self.de.density_score(self.X_pool_index)
            ranking = ds * isli
            
        if self.selection == QueryStrategy.isli:            
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            isli = self.ie.isli()
            ranking =  isli
            
        if self.selection == QueryStrategy.isls_density:
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            isls = self.ie.isls()
            ds = self.de.density_score(self.X_pool_index)
            ranking = ds * isls
            
        if self.selection == QueryStrategy.isls:
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            isls = self.ie.isls()
            ranking = isls  
            
        if self.selection == QueryStrategy.unc_isls:
            self.ie.next_iteration(self.X_pool_index, ranking, proba_pool.argmax(axis=1))
            islsq = self.ie.islsq()
            ranking = ranking * islsq    
   
        return (-ranking).argsort()[:step_size]

    
##########################################
##########################################
##########################################

class SKLearnActiveLearnerSparse(ActiveLearner):
    
    def __init__(self, warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, Model, model_args = {}, selection = QueryStrategy.unc):
        self.Model = Model
        self.model_args = model_args
        super().__init__(warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, selection)
        
    def _train(self):
        self.n_train = self.n_train + 1
        X = scipy.sparse.vstack(self.X_al_training)
        self.clf = self.Model(**self.model_args)
        
        # training
        train_start = timer()
        self.clf.fit(X, self.y_al_training)
        train_end = timer()
        time_train = train_end - train_start
        
        # inference
        inf_start = timer()
        proba = self.clf.predict_proba(self.X_test)
        y_pred = proba.argmax(axis=1)
        inf_end = timer()
        time_inf = inf_end - inf_start
        
        f1_mic, f1_mac, precision, recall, c = self._print_metrics(y_pred, time_train, time_inf)
        return f1_mic, f1_mac, precision, recall, c, time_train, time_inf
     
    def _query(self, X_pool, step_size):
        X = scipy.sparse.vstack(self.X_al_training)
        proba_pool = self.clf.predict_proba(X)
        unc = least_confidence(proba_pool)
       
        return (-unc).argsort()[:step_size]

    
##########################################
##########################################
##########################################

def getModelMLP(input_shape, n_classes, initial_weights=None):
    input_layer = tfk.Input(shape=(input_shape,), dtype='float32')

    x = tfk.layers.Dense(500, activation='relu')(input_layer)
    x = tfk.layers.Dropout(0.5)(x)
    x = tfk.layers.Dense(500, activation='relu')(x)
    x = tfk.layers.Dropout(0.5)(x)

    output_layer = tfk.layers.Dense(n_classes, activation='softmax')(x)
    model = tfk.Model(input_layer, output_layer)
    
    if initial_weights:
        model.set_weights(initial_weights)
    
    return model

def getModel_Embedding(initial_weights=None):
    
    vocab_size = 5000
    embedding_dim = 50
    max_length = 400
            
    model = tfk.Sequential([
        tfk.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tfk.layers.GlobalAveragePooling1D(),
        #tfk.layers.Dense(10, activation='relu'),
        tfk.layers.Dense(2, activation='softmax')
    ])
    
    if initial_weights:
        model.set_weights(initial_weights)
    return model

def getModel_Embedding_pre(embedding_layer, initial_weights=None):
    
    from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Sequential
    n_classes = 2
    import tensorflow as tf

    l2_reg = 0.00001
    nb_feature_maps = 128

    model = Sequential()
    model.add(embedding_layer)
    model.add(tfk.layers.GlobalAveragePooling1D())
    #model.add(tfk.layers.Dense(100, activation='relu'))
    #model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))

    
    if initial_weights:
        model.set_weights(initial_weights)
    return model

def getModel_KimCNN(embedding_layer, initial_weights=None):
    
    from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Sequential
    n_classes = 2
    import tensorflow as tf

    l2_reg = 0.00001
    nb_feature_maps = 128

    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.4))
    model.add(Conv1D(nb_feature_maps,
                     3,
                     padding='valid',
                     activation='relu',
                     kernel_regularizer=l2(l2_reg)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))

    
    if initial_weights:
        model.set_weights(initial_weights)
    return model

##########################################
##########################################
##########################################

def mcd_infer(model, data, T):
    if T > 1:
        data = np.array(data)

        f = K.function([model.layers[0].input], [model.layers[-1].output])
        with K.eager_learning_phase_scope(value=0 if T == 1 else 1):  # 0=test, 1=train
            mcd_output =  [f(data) for _ in range(T)]
        mcd_output = np.array(mcd_output)
        p_mean = (np.mean(mcd_output, axis=0)[0])
        return np.array(p_mean)
    else:
        preds = model.predict(data)
        return np.array(preds)


Word2VecModelPath = '/home/jovyan/moderated_classifier/GoogleNews-vectors-negative300.bin.gz'

MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 400

def get_embeddings_index(model):
    embeddings_index = model.wv.vocab
    
    for word, vocab in embeddings_index.items():
        embeddings_index[word] = model.wv.vectors[vocab.index]
    return embeddings_index, model.vector_size

def get_embedding_index_glove():
    path_to_glove_file = "./glove.6B.50d.txt"

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index, 50

def get_embedding_layer(word_index, embedding_index, embedding_dim, static=False):
    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words+1, embedding_dim))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return tfk.layers.Embedding(num_words+1,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=static)


from enum import Enum, auto
class Model(Enum):
    MLP = auto()
    TrainableWordEmbeddings = auto()
    PreTrainedGloVeEmbeddings = auto()
    KimCNNWord2Vec = auto()
    PreTrainedWordEmbeddings = auto()

class KerasActiveLearner(ActiveLearner):
    
    def __init__(self, model, warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index, mcd=False, model_args=None):
        self.mcd = mcd
        self.type = model
        
        if self.type is Model.MLP:
            self.model = getModelMLP(768, 2)
        elif self.type is Model.TrainableWordEmbeddings:
            self.model = getModel_Embedding()
        elif self.type is Model.PreTrainedGloVeEmbeddings:
            embeddings_index, embedding_dim = get_embedding_index_glove()
            self.embedding_layer = get_embedding_layer(model_args['word_index'], embeddings_index, embedding_dim, True)
            self.model = getModel_Embedding_pre(self.embedding_layer)
            
        elif self.type is Model.PreTrainedWordEmbeddings:
            w = models.KeyedVectors.load_word2vec_format(Word2VecModelPath, binary=True)    
            embeddings_index, embedding_dim = get_embeddings_index(w)
            w = None

            self.embedding_layer = get_embedding_layer(model_args['word_index'], embeddings_index, embedding_dim)
            self.model = getModel_Embedding_pre(self.embedding_layer)
            
            
        elif self.type is Model.KimCNNWord2Vec:
            w = models.KeyedVectors.load_word2vec_format(Word2VecModelPath, binary=True)    
            embeddings_index, embedding_dim = get_embeddings_index(w)
            w = None

            self.embedding_layer = get_embedding_layer(model_args['word_index'], embeddings_index, embedding_dim)
            self.model = getModel_KimCNN(self.embedding_layer)
        else:
            print('No model found')
        
        self.initial_weights = copy.deepcopy(self.model.get_weights())
        super().__init__(warmstart_size, X_pool, y_pool, X_test, y_test, X_pool_index)
    
    def get_initial_model(self):
        if self.type is Model.MLP:
            model = getModelMLP(768, 2, self.initial_weights)
        elif self.type is Model.TrainableWordEmbeddings:
            model = getModel_Embedding(self.initial_weights)
        elif self.type is Model.PreTrainedGloVeEmbeddings:
            model = getModel_Embedding_pre(self.embedding_layer)
        elif self.type is Model.PreTrainedWordEmbeddings:
            model = getModel_Embedding_pre(self.embedding_layer)
        elif self.type is Model.KimCNNWord2Vec:
            model = getModel_KimCNN(self.embedding_layer)
            
        return model
    
    def _train(self):
        self.n_train = self.n_train + 1
        del self.model
        
        # 
        if self.type is Model.MLP:
            self.model = getModelMLP(768, 2, self.initial_weights)
        elif self.type is Model.TrainableWordEmbeddings:
            self.model = getModel_Embedding(self.initial_weights)
        elif self.type is Model.PreTrainedGloVeEmbeddings:
            self.model = getModel_Embedding_pre(self.embedding_layer)
        elif self.type is Model.PreTrainedWordEmbeddings:
            self.model = getModel_Embedding_pre(self.embedding_layer)
        elif self.type is Model.KimCNNWord2Vec:
            self.model = getModel_KimCNN(self.embedding_layer)
        #   
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        callback = tfk.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        
        # training
        train_start = timer()
        history = self.model.fit(
            np.array(self.X_al_training), 
            to_categorical(self.y_al_training), 
            batch_size=32, 
            epochs=10,
            validation_data=(np.array(self.X_test), to_categorical(self.y_test)),
            callbacks=[callback],
            #shuffle=True,
            verbose=0,
        )
        train_end = timer()
        time_train = train_end - train_start
        
        if False:
            from matplotlib import pyplot as plt
            print(history.history.keys())
            #  "Accuracy"
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            # "Loss"
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        
        # inference
        inf_start = timer()
        p_mean = mcd_infer(self.model, self.X_test, 50 if self.mcd else 1)
        y_pred = p_mean.argmax(axis=1)
        inf_end = timer()
        time_inf = inf_end - inf_start
        
        f1, precision, recall, c = self._print_metrics(y_pred, time_train, time_inf)
        return f1, precision, recall, c, time_train, time_inf
        
    
    def _query(self, X_pool, step_size):
        p_mean = mcd_infer(self.model, X_pool, 50 if self.mcd else 1)
        
        unc = 1 - np.array(p_mean).max(axis=1)
        #unc = - np.sum(p_mean * np.log(p_mean + 1e-10), axis=-1)
        return (-unc).argsort()[:step_size]

##########################################
##########################################
##########################################

