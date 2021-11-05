from sklearn.utils import shuffle
import pandas as pd
import re
import numpy as np
import os

n = 50000

def _load_hate_speech():
    data = pd.read_csv("./datasets/train.csv")
    doc_labels = (data['toxic'] == 1) | (data['severe_toxic']==1) | (data['obscene']==1) | (data['threat']==1) | (data['insult']==1) | (data['identity_hate']==1)
    df = pd.DataFrame ({'id': data.id, 'comment_text': data.comment_text, 'label' : doc_labels.map({True: 1, False: 0})})
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    df_sort = df.sort_values('label')
    df_sort = df_sort.iloc[-n: , :]
    
    X = df_sort['comment_text'].tolist()
    y = df_sort['label'].tolist()
    
    return X, y

# to remove IPs
def remove_IP(text):
    url = re.compile(r'[0-9]+(?:\.[0-9]+){3}')
    return url.sub(r'',text)

# to remove URLs
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# to remove html tags
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

if __name__ == "__main__":
    print('Prepare IMDB dataset ...')
    
    if not (os.path.isfile('./datasets/hate_speech_X.npy') and os.path.isfile('./datasets/hate_speech_X.npy')): 
        X, y = _load_hate_speech()
        X, y = shuffle(X, y, random_state = 42)
        
        df = pd.DataFrame(data=(X))
        df[0] = df[0].apply(remove_IP)
        df[0] = df[0].apply(remove_URL)
        df[0] = df[0].apply(remove_html)

        np.save('./datasets/hate_speech_X', df[0].tolist())
        np.save('./datasets/hate_speech_y', y)
        #np.save('imdb_index', [i for i in range(len(X))])
        
        print('Done')
    else:
        print('hate_speech dataset already exists.')
    