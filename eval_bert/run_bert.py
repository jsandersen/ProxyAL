import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# python run_al_experiment.py --dataset app_store --steps 100 --encoding bert --save_repeat 5 --query_strategy unc --random 

import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/../.') 

import math
import argparse
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from src.EvaluationLoop import EvaluationLoop
from src.training_loop2 import QueryStrategy
from data.datasets import Datasets, Encodings
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from transformers import TFBertForSequenceClassification, TFTrainer, TFTrainingArguments, BertTokenizerFast

parser = argparse.ArgumentParser(description='Run BERT')
parser.add_argument("--name", type=str, default='test')
parser.add_argument("--dataset", type=Datasets.from_string, choices=list(Datasets))
parser.add_argument("--encoding", type=Encodings.from_string, choices=list(Encodings))
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--save_repeat", type=int, default=1)
parser.add_argument('--query_strategy', type=QueryStrategy.from_string, choices=list(QueryStrategy))
parser.add_argument('--random', default=False, action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    name = args.name
    dataset = args.dataset
    encoding = args.encoding
    al_steps = args.steps
    save_and_repeat = args.save_repeat
    sample_size_per_step = args.batch_size
    query_strategy = args.query_strategy
    random = args.random
    
    print(args)
    
    name_postfix = ""
    
    print('#############')
    print(random)
    print('#############')
    
    if random:
        print('Random query strategy is used. Selection is ignored.')
        name_postfix = "_rnd"
        query_strategy = QueryStrategy.unc
    
    print('load index ... ')
    used_training_index_dir = f'../eval_proxy/res/{dataset}_{encoding}_{str(query_strategy)}_{name}{name_postfix}_{sample_size_per_step}_{al_steps}_{save_and_repeat}/used_training_index.npy'
    
    print(used_training_index_dir)
    used_training_index = np.load(used_training_index_dir, allow_pickle=True)
    
    print('load data...')
    data_X = np.load(f'../data/datasets/{dataset}_X.npy')
    data_y = np.load(f'../data/datasets/{dataset}_y.npy')
    
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    sss.get_n_splits(data_X, data_y)
    i = 0
    for train_index, test_index in sss.split(data_X, data_y):
        print(f'Model #{i} ...')
        X_train, X_test = data_X[train_index], data_X[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]
        
        idx = used_training_index[i][-1]
        
        X_used = X_train[idx]
        y_used = y_train[idx]
        
        X_bert_train, X_bert_eval, y_bert_train, y_bert_eval = train_test_split(X_used, y_used, test_size=0.10, random_state=42, stratify=y_used)
        
        print('tokenize ...')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        train_encodings = tokenizer(X_bert_train.tolist(), truncation=True, padding=True)
        val_encodings = tokenizer(X_bert_eval.tolist(), truncation=True, padding=True)
        test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)
        
        del X_bert_train
        del X_bert_eval
        del X_test
        del tokenizer

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            y_bert_train
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            y_bert_eval
        ))
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            y_test
        ))
        
        del train_encodings
        del val_encodings
        del test_encodings
        
        training_args = TFTrainingArguments(
                    evaluation_strategy= "steps",
                    output_dir='./bert_models',  
                    num_train_epochs=5,              # total number of training epochs
                    per_device_train_batch_size=16,  # batch size per device during training
                    per_device_eval_batch_size=32,   # batch size for evaluation
                    #warmup_steps=500,                # number of warmup steps for learning rate scheduler
                    weight_decay=0.01,               # strength of weight decay
                    warmup_steps=math.ceil(len(y_bert_train) / 16),
                    logging_dir='./logs',         # directory for storing logs
                    logging_steps=2000,
                    learning_rate= 5e-5,
                    save_total_limit = 1,
                    #load_best_model_at_end= True,
                )
        
        print('Load BERT model ...')
        
        with training_args.strategy.scope():
            model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

        trainer = TFTrainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset             # evaluation dataset
        )
        
        print('Train BERT model ...')
        trainer.train()
        
        print('Predict ...')
        preds = trainer.model.predict(test_dataset.batch(8))
        p_softmax = tf.nn.softmax(preds.logits, axis=1).numpy()
        y_pred = p_softmax.argmax(axis=1)

        save_dir = f'./res/{dataset}_{encoding}_{str(query_strategy)}_{name}{name_postfix}_{sample_size_per_step}_{al_steps}_{save_and_repeat}'
        
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        np.save(f'{save_dir}/y_pred_{i}', y_pred)
        np.save(f'{save_dir}/y_true_{i}', y_test)
        
        print(f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))

        del trainer
        del train_dataset
        del val_dataset
        del test_dataset
        del model
        del preds
        del p_softmax
        del y_pred
        del training_args
        
        i = i + 1
    
    